"""
SVG Reshape V2 - Batch SVG Class Cleanup & Filename Normalization

Lightweight preprocessing step that cleans up SVG floorplan files without
modifying any geometry, viewBox, or layout. For full rescaling/recentering,
use svg_reshape.py (v1) instead.

Key Operations:
    1. Class Removal (convert_unnecessary_to_background):
       Elements whose semanticId maps to unnecessary furniture/fixture
       categories (beds, wardrobes, bath tubs, etc.) have their semanticId
       and instanceId stripped, converting them to unlabeled background.

    2. Stroke Width Normalization:
       Sets stroke-width to 0.1 on all path, circle, and ellipse elements
       that have a stroke-width attribute.

    3. Output Filename Normalization:
       Strips suffixes like '_predicted_editted.svg' and '_predicted.svg'
       back to plain '.svg' for clean output filenames.

Usage:
    python svg_reshape_v2.py \\
        --input  <input_dir> \\
        --output <output_dir> \\
        --ignore_mode poilabs \\
        --nproc 64 \\
        --n_samples 100 --seed 42
"""

import os
import xml.etree.ElementTree as ET
import argparse
import glob
import mmcv
import random

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

CONSTANT_STROKE_WIDTH = "0.1"

# Ignore-label modes: which 0-indexed class IDs to strip
IGNORE_LABEL_MODES = {
    "background_only": [35],
    "poilabs": [35, 3, 5, 7, 8, 9, 11, 14, 15, 17, 19, 20, 21, 22, 23],
}


def _ns(tag):
    return f"{{{SVG_NS}}}{tag}"


def should_convert_to_background(elem, ignore_labels):
    """
    Check if element should be converted to background based on semanticId.
    Returns True if element's class is in ignore_labels.
    """
    semantic_id = elem.attrib.get("semanticId")

    if semantic_id is None:
        return False

    try:
        # SVG uses 1-based indexing, convert to 0-based
        sem_id = int(semantic_id) - 1
        return sem_id in ignore_labels
    except ValueError:
        return False


def convert_unnecessary_to_background(root, ignore_labels):
    """
    Convert elements with ignored classes to background.
    Removes both semanticId and instanceId attributes entirely.
    Returns the number of elements converted.
    """
    converted_count = 0

    # Find all groups
    groups = root.findall(".//" + _ns("g"))

    # If no groups, work directly with root
    if not groups:
        groups = [root]

    # Process elements in each group
    for group in groups:
        for elem in list(group):
            tag = elem.tag
            # Only process drawing elements
            if any(tag.endswith(t) for t in ["path", "circle", "ellipse"]):
                if should_convert_to_background(elem, ignore_labels):
                    # Convert to background: remove both semanticId and instanceId
                    if "semanticId" in elem.attrib:
                        del elem.attrib["semanticId"]
                    if "instanceId" in elem.attrib:
                        del elem.attrib["instanceId"]
                    converted_count += 1

    return converted_count


def normalize_stroke_widths(root):
    """
    Set stroke-width to CONSTANT_STROKE_WIDTH on all path, circle, and
    ellipse elements that have a stroke-width attribute.
    Returns the number of elements updated.
    """
    updated_count = 0

    for tag_name in ["path", "circle", "ellipse"]:
        for elem in root.findall(".//" + _ns(tag_name)):
            if elem.get("stroke-width") is not None:
                elem.set("stroke-width", CONSTANT_STROKE_WIDTH)
                updated_count += 1

    return updated_count


def process(svg_file, output_dir, ignore_labels):
    """
    Process a single SVG file:
      1. Remove classes specified by ignore_labels
      2. Normalize stroke widths to 0.1
      3. Write with cleaned-up filename
    """
    # Normalize output filename
    svg_filename = os.path.basename(svg_file)
    if svg_filename.endswith("_predicted_editted.svg"):
        svg_filename = svg_filename.replace("_predicted_editted.svg", ".svg")
    elif svg_filename.endswith("_pred_editted.svg"):
        svg_filename = svg_filename.replace("_pred_editted.svg", ".svg")
    elif svg_filename.endswith("_predicted.svg"):
        svg_filename = svg_filename.replace("_predicted.svg", ".svg")
    svg_out_path = os.path.join(output_dir, svg_filename)

    # Parse SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Remove unnecessary classes
    converted_count = convert_unnecessary_to_background(root, ignore_labels)

    # Normalize stroke widths
    stroke_count = normalize_stroke_widths(root)

    # Write output
    os.makedirs(os.path.dirname(svg_out_path), exist_ok=True)
    tree.write(svg_out_path)

    print(
        f"[clean] {os.path.basename(svg_file)} -> {svg_filename}  "
        f"classes_removed={converted_count}, strokes_set={stroke_count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVG class cleanup & filename normalization (no geometry changes)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing SVG files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed SVG files",
    )
    parser.add_argument(
        "--nproc", type=int, default=64, help="Number of parallel processes"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of random samples to process (if not set, process all files)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--ignore_mode",
        type=str,
        default="poilabs",
        choices=list(IGNORE_LABEL_MODES.keys()),
        help="Which classes to remove: "
        + ", ".join(f"'{k}' ({len(v)} classes)" for k, v in IGNORE_LABEL_MODES.items())
        + " (default: poilabs)",
    )

    args = parser.parse_args()

    ignore_labels = set(IGNORE_LABEL_MODES[args.ignore_mode])

    data_dir = args.input
    output_dir = args.output

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    svg_paths = sorted(glob.glob(os.path.join(data_dir, "*.svg")))

    # Random sampling if --n_samples is provided
    if args.n_samples is not None and args.n_samples > 0:
        if args.n_samples >= len(svg_paths):
            print(
                f"Warning: --n_samples ({args.n_samples}) >= total files ({len(svg_paths)}). Using all files."
            )
        else:
            random.seed(args.seed)
            svg_paths = random.sample(svg_paths, args.n_samples)
            print(f"Randomly sampled {args.n_samples} files (seed={args.seed})")

    if len(svg_paths) == 0:
        print(f"No SVG files found in {data_dir}")
    else:
        print(f"Found {len(svg_paths)} SVG files. Processing in parallel...")
        print(
            f"Ignore mode: '{args.ignore_mode}' -> removing {len(ignore_labels)} class IDs: {sorted(ignore_labels)}"
        )

        def process_wrapper(svg_file):
            return process(svg_file, output_dir, ignore_labels)

        mmcv.track_parallel_progress(process_wrapper, svg_paths, args.nproc)

        print(f"Processing complete. Output saved to {output_dir}")
