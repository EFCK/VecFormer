import os
import numpy as np
import xml.etree.ElementTree as ET
from svgpathtools import parse_path, svgstr2paths
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

from model.vecformer.evaluator.evaluator import Evaluator, EvaluatorConfig
from utils.svg_util import get_namespace, add_ns, del_ns, primitive2str


# FloorPlanCAD 36-class color table (0-indexed; index == VecFormer label).
# Thing classes: indices 0-29 (ids 1-30). Stuff classes: indices 30-34 (ids 31-35).
# Background: index 35 (id 36).
SVG_CATEGORIES = [
    {"color": [224,  62, 155], "isthing": 1, "id":  1, "name": "single door"},
    {"color": [157,  34, 101], "isthing": 1, "id":  2, "name": "double door"},
    {"color": [232, 116,  91], "isthing": 1, "id":  3, "name": "sliding door"},
    {"color": [101,  54,  72], "isthing": 1, "id":  4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id":  5, "name": "revolving door"},
    {"color": [142,  76, 101], "isthing": 1, "id":  6, "name": "rolling door"},
    {"color": [ 96,  78, 245], "isthing": 1, "id":  7, "name": "window"},
    {"color": [ 26,   2, 219], "isthing": 1, "id":  8, "name": "bay window"},
    {"color": [ 63, 140, 221], "isthing": 1, "id":  9, "name": "blind window"},
    {"color": [233,  59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [ 94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [ 66, 107,  81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [ 94, 150,  83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [ 66, 107,  59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151,  82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [ 81, 107,  58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152,  83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [ 80,  72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108,  59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    {"color": [247, 206,  75], "isthing": 1, "id": 28, "name": "stairs"},
    {"color": [237, 112,  45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233,  59,  46], "isthing": 1, "id": 30, "name": "escalator"},
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102,  67,  62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167,  92,  32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [ 64,  52, 105], "isthing": 0, "id": 35, "name": "railing"},
    {"color": [  0,   0,   0], "isthing": 0, "id": 36, "name": "bg"},
]

IGNORE_LABEL_MODES = {
    "background_only": [35],
    "poilabs": [35, 3, 5, 7, 8, 9, 11, 14, 15, 17, 19, 20, 21, 22, 23],
}

THING_IDXS = list(range(30))
STUFF_IDXS = list(range(30, 35))


def resolve_svg_path(npy_filepath: str, svg_dir: str = None) -> str:
    """Convert stored filepath to actual on-disk SVG path.

    Auto-remaps: .../line_json/<folder>/file.svg -> .../svg/<folder>/file.svg
    Override with svg_dir to specify root explicitly.
    """
    path = npy_filepath.replace('\\', '/')
    if svg_dir:
        parts = path.split('/line_json/')
        rel = parts[1] if len(parts) == 2 else os.path.basename(path)
        return os.path.join(svg_dir, rel)
    return path.replace('/line_json/', '/svg/')


def parse_svg_tree(svg_path):
    """Parse SVG and return (tree, root, namespace) for in-place editing."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    namespace = get_namespace(root)
    return tree, root, namespace


def get_drawable_primitives(root, namespace):
    """Yield drawable primitives in the exact same order as preprocess.py parse_svg()."""
    groups = list(root.iter(add_ns("g", namespace)))
    if len(groups) > 0:
        for group in groups:
            if len(group) == 0:
                continue
            for primitive in group:
                yield primitive
    else:
        drawable_tags = {add_ns(t, namespace) for t in
            ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")}
        for child in root:
            if child.tag in drawable_tags:
                yield child


def compute_primitive_length(primitive, namespace):
    """Compute primitive length exactly like preprocess.py parse_primitive().
    Returns length (float). Returns 0.0 if primitive is invalid/degenerate."""
    tag = del_ns(primitive.tag, namespace)
    try:
        if tag == "path":
            path_str = primitive.attrib.get("d", "")
            if not path_str:
                return 0.0
            path = parse_path(path_str)
            length = path.length()
            return length if isinstance(length, float) else 0.0
        else:
            # rect, circle, ellipse, line, polyline, polygon
            paths, _ = svgstr2paths(primitive2str(primitive))
            if not paths:
                return 0.0
            length = paths[0].length()
            return length if isinstance(length, float) else 0.0
    except Exception:
        return 0.0

def visualSVG_with_ids(tree, root, namespace, sem_labels, ins_labels, out_path, ignore_labels=None, cvt_color=False):
    """
    Visualize SVG with model predictions for both semantic and instance IDs.

    Uses the same traversal order as preprocess.py parse_svg() to keep primitive
    indices in sync with model_output.npy.

    Args:
        tree: ElementTree from parse_svg_tree()
        root: root Element from parse_svg_tree()
        namespace: SVG namespace from parse_svg_tree()
        sem_labels: semantic labels array (N_primitives,)
        ins_labels: instance labels array (N_primitives,)
        out_path: output SVG path
        ignore_labels: List of label indices to filter out (map to background).
        cvt_color: whether to set white background
    """
    if ignore_labels is None:
        ignore_labels = []

    if cvt_color:
        root.attrib["style"] = "background-color: #255255255;"

    ind = 0
    for primitive in get_drawable_primitives(root, namespace):
        # Check validity exactly like preprocess.py
        length = compute_primitive_length(primitive, namespace)
        if length < 1e-10:
            continue  # Skip without incrementing — preprocess also skips

        # Sanity check
        if ind >= len(sem_labels):
            tag = del_ns(primitive.tag, namespace)
            raise IndexError(
                f"Primitive index out of bounds: ind={ind} >= len(sem_labels)={len(sem_labels)}, "
                f"tag={tag}, file={out_path}")

        sem_label = sem_labels[ind]
        ins_label = ins_labels[ind]

        # Apply ignore labels
        if sem_label in ignore_labels:
            sem_label = 35  # Map to background
            primitive.attrib.pop("semanticId", None)
            primitive.attrib.pop("instanceId", None)
        else:
            primitive.attrib["semanticId"] = str(int(sem_label) + 1)  # +1: SVG uses 1-based
            if sem_label in range(30, 35):
                primitive.attrib["instanceId"] = "-1"
            else:
                primitive.attrib["instanceId"] = str(int(ins_label))

        # Set visualization color
        color = SVG_CATEGORIES[sem_label]["color"]
        primitive.attrib["stroke"] = "rgb({:d},{:d},{:d})".format(color[0], color[1], color[2])
        primitive.attrib["fill"] = "none"
        primitive.attrib["stroke-width"] = "0.2"

        ind += 1

    # Sanity check: did we consume all labels?
    if ind != len(sem_labels):
        print(f"WARNING: primitive count mismatch in {out_path}: "
              f"visited {ind} primitives but have {len(sem_labels)} labels")

    # Write modified tree directly — preserves <g> hierarchy
    tree.write(out_path, xml_declaration=True, encoding="unicode")
    return out_path


def process_dt_with_ids(input):
    svg_path, sem_labels, ins_labels, out_path, png_out_path, generate_png, coords, ignore_labels = input

    tree, root, namespace = parse_svg_tree(svg_path)
    visualSVG_with_ids(tree, root, namespace, sem_labels, ins_labels, out_path, ignore_labels=ignore_labels)

    if generate_png:
        # First convert SVG to PNG
        svg2png(out_path, png_out_path, background_color="white", scale=7)

        # Then draw bounding boxes and IDs on the PNG
        draw_bboxes_and_ids(png_out_path, coords, sem_labels, ins_labels, scale=7)


def svg2png(svg_path, png_path, background_color="white", scale=7.0):
    '''
    Convert svg to png
    '''
    command = "cairosvg {} -o {} -b {} -s {}".format(svg_path, png_path, background_color, scale)
    os.system(command)

def draw_bboxes_and_ids(png_path, coords, sem_labels, ins_labels, scale=7.0):
    """
    Draw bounding boxes and instance IDs on PNG image
    """
    image = Image.open(png_path)
    draw = ImageDraw.Draw(image, 'RGBA')

    # Get original coords shape (N, 4, 2)
    if len(coords.shape) == 3:
        coords_2d = coords
        num_points = coords.shape[0]
    elif len(coords.shape) == 2 and coords.shape[1] == 8:
        num_points = coords.shape[0]
        coords_2d = coords.reshape(-1, 4, 2)
    else:
        return

    # Trim labels to match actual data points
    num_actual_points = min(num_points, len(sem_labels), len(ins_labels))
    sem_labels_trimmed = sem_labels[:num_actual_points]
    ins_labels_trimmed = ins_labels[:num_actual_points]

    # Get unique semantic-instance pairs
    labels = np.concatenate([sem_labels_trimmed[:, None], ins_labels_trimmed[:, None]], axis=1)
    uni_labels = np.unique(labels, axis=0)

    for ulabel in uni_labels:
        sem, ins = ulabel

        # Skip background or stuff classes
        if sem >= 30 or ins < 0:
            continue

        # Get mask for this instance
        mask = np.logical_and(labels[:, 0] == sem, labels[:, 1] == ins)

        # Get coordinates for this instance
        inst_coords = coords_2d[mask].reshape(-1, 2)

        if len(inst_coords) == 0:
            continue

        # Calculate bounding box
        x1, y1 = np.min(inst_coords[:, 0], axis=0), np.min(inst_coords[:, 1], axis=0)
        x2, y2 = np.max(inst_coords[:, 0], axis=0), np.max(inst_coords[:, 1], axis=0)

        # Get color for this semantic class
        color = tuple(SVG_CATEGORIES[int(sem)]["color"])

        # Draw bounding box with transparency
        draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                       fill=color + (32,), outline=color, width=2)

        # Draw text with instance ID only
        text = f'{ins}'
        draw.text((x1 * scale, y1 * scale), text, fill=(0, 0, 0), align='left')

    # Save the modified image
    image.save(png_path)



def get_path(root, namespace):
    """Extract path coordinates from SVG using the same traversal as preprocess.py.

    Args:
        root: root Element from parse_svg_tree()
        namespace: SVG namespace from parse_svg_tree()

    Returns:
        widths, gids, args, lengths, types
    """
    args, widths, gids, lengths, types = [], [], [], [], []
    COMMANDS = ['Line', 'Arc', 'circle', 'ellipse']

    for primitive in get_drawable_primitives(root, namespace):
        length = compute_primitive_length(primitive, namespace)
        if length < 1e-10:
            continue

        tag = del_ns(primitive.tag, namespace)
        widths.append(primitive.attrib.get("stroke-width", "0.1"))
        gid = int(primitive.attrib["gid"]) if "gid" in primitive.attrib else -1
        gids.append(gid)

        # Sample 4 points along the path for coordinate extraction
        sample_ts = [0, 1/3, 2/3, 1.0]
        arg = []

        if tag == "path":
            path_repre = parse_path(primitive.attrib.get("d", ""))
            try:
                for t in sample_ts:
                    point = path_repre.point(t)
                    arg.extend([point.real, point.imag])
            except (RuntimeError, ValueError):
                try:
                    start_point = path_repre[0].start
                    for _ in sample_ts:
                        arg.extend([start_point.real, start_point.imag])
                except (AttributeError, IndexError):
                    for _ in sample_ts:
                        arg.extend([0.0, 0.0])
            path_type = path_repre[0].__class__.__name__
            types.append(COMMANDS.index(path_type) if path_type in COMMANDS else 0)
        else:
            # All non-path types: convert via svgstr2paths (same as preprocess)
            try:
                paths, _ = svgstr2paths(primitive2str(primitive))
                path_obj = paths[0]
                for t in sample_ts:
                    point = path_obj.point(t)
                    arg.extend([point.real, point.imag])
                path_type = path_obj[0].__class__.__name__
                types.append(COMMANDS.index(path_type) if path_type in COMMANDS else 0)
            except Exception:
                for _ in sample_ts:
                    arg.extend([0.0, 0.0])
                types.append(0)

        args.append(arg)
        lengths.append(length)

    return widths, gids, args, lengths, types


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import tqdm

    parser = argparse.ArgumentParser(description='SVG Visualization with Model Predictions')
    parser.add_argument('--res_file', type=str, default="./results/floorplancad/",
                        help='Path to the results directory (containing model_output.npy)')
    parser.add_argument('--svg_dir', type=str, default="",
                        help='Optional root directory for SVG files (overrides auto path remap)')
    parser.add_argument('--min_score', type=float, default=0.1,
                        help='Minimum confidence score for instance predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for panoptic quality evaluation')
    parser.add_argument('--generate_png', action='store_true', default=False,
                        help='Generate PNG files from SVG (requires cairosvg)')
    parser.add_argument('--semantic', action='store_true', default=False,
                        help='Use semantic scores from semantic predictions instead of instance scores')
    parser.add_argument('--out_dir', type=str, default="",
                        help='Path to the output directory')
    parser.add_argument('--perfile', action='store_true', default=False,
                        help='Generate per-file PQ/RQ/SQ results alongside aggregated results')
    parser.add_argument('--eval_mode', type=str, default="original",
                        choices=["original", "strict"],
                        help="Evaluation mode: 'original' uses target-centric matching (no dedup), "
                             "'strict' uses prediction-centric greedy matching with dedup and orphan FP counting")
    parser.add_argument('--ignore_mode', type=str, default="background_only",
                        choices=["none"] + list(IGNORE_LABEL_MODES.keys()),
                        help="Ignore label mode: 'none' evaluates all classes, "
                             "'background_only' ignores background (class 35), "
                             "'poilabs' ignores background + classes not in Poilabs data")
    args = parser.parse_args()

    # Resolve ignore labels based on --ignore_mode
    if args.ignore_mode == "none":
        ignore_labels = []
    else:
        ignore_labels = IGNORE_LABEL_MODES[args.ignore_mode]

    min_score = args.min_score
    iou_threshold = args.iou_threshold
    svg_dir = args.svg_dir if args.svg_dir else None

    eval_mode = args.eval_mode
    print(f"Ignore label mode: {args.ignore_mode} ({ignore_labels})")
    print(f"Eval mode: {eval_mode}")

    # Create VecFormer evaluator
    evaluator = Evaluator(EvaluatorConfig(
        num_classes=35,
        ignore_label=ignore_labels,
        iou_threshold=iou_threshold,
        output_dir="",
    ))

    # Global PQ accumulators
    total_tp = torch.zeros(35)
    total_fp = torch.zeros(35)
    total_fn = torch.zeros(35)
    total_tp_iou = torch.zeros(35, dtype=torch.float32)

    # Valid class indices for PQ computation
    valid_thing = [i for i in THING_IDXS if i not in ignore_labels]
    valid_stuff = [i for i in STUFF_IDXS if i not in ignore_labels]
    valid_all = valid_thing + valid_stuff

    _EPS = torch.finfo(torch.float32).eps

    def cal_scores(tp, fp, fn, tp_iou):
        rq = tp / (tp + 0.5 * fn + 0.5 * fp + _EPS)
        sq = tp_iou / (tp + _EPS)
        pq = rq * sq
        return pq.item() * 100, sq.item() * 100, rq.item() * 100

    res_file = Path(args.res_file) / "model_output.npy"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.res_file)
    generate_png = args.generate_png
    semantic = args.semantic
    perfile = args.perfile
    os.makedirs(out_dir, exist_ok=True)

    perfile_results = []

    detections = np.load(res_file, allow_pickle=True)
    inputs = []
    coco_res = []

    for det in tqdm.tqdm(detections):
        svg_path = resolve_svg_path(det["filepath"], svg_dir)
        if not os.path.exists(svg_path):
            print(f"Warning: SVG not found, skipping: {svg_path}")
            continue

        tree, root, namespace = parse_svg_tree(svg_path)
        widths, gids, path_args, lengths_arr, types = get_path(root, namespace)
        widths = np.array(widths)
        gids = np.array(gids)
        lengths_arr = np.array(lengths_arr)
        types = np.array(types)
        coords = np.array(path_args).reshape(-1, 4, 2)

        ins_outs = det["ins"]
        semantic_bits = det["sem"]

        # Determine number of primitives
        if len(ins_outs):
            shape = ins_outs[0]["masks"].shape[0]
        else:
            shape = semantic_bits.shape[0]

        if semantic:
            # Suppress background: argmax over classes 0-34 only (never predicts class 35)
            sem_out = np.argmax(semantic_bits[:, :35], axis=1).astype(np.int64)
            ins_out = np.full(shape, -1, dtype=np.int64)
            instances = []
            for instance in ins_outs:
                if instance["scores"] < min_score:
                    continue
                masks, labels = instance["masks"], instance["labels"]
                sem_out[masks] = labels
                ins_out[masks] = len(instances)
                label_for_inst = int(sem_out[masks][0]) if masks.any() else int(labels)
                instances.append({
                    "masks": masks,
                    "labels": label_for_inst,
                    "scores": instance["scores"],
                })
        else:
            sem_out = np.full(shape, 35, dtype=np.int64)
            ins_out = np.full(shape, -1, dtype=np.int64)
            instances = []
            for instance in ins_outs:
                if instance["scores"] < min_score:
                    continue
                masks, labels = instance["masks"], instance["labels"]
                sem_out[masks] = labels
                ins_out[masks] = len(instances)
                instances.append({
                    "masks": masks,
                    "labels": labels,
                    "scores": instance["scores"],
                })

        # Filter out unnecessary classes (ignored non-background labels)
        unnecessary = [l for l in ignore_labels if l != 35]
        for label in unnecessary:
            mask = sem_out == label
            sem_out[mask] = 35
            ins_out[mask] = -1
        if unnecessary:
            instances = [inst for inst in instances
                         if (sem_out[inst["masks"]] != 35).any()]

        # Build panoptic predictions for VecFormer evaluator
        pred_masks_list, pred_labels_list = [], []
        for inst in instances:
            pred_masks_list.append(torch.from_numpy(inst["masks"]))
            pred_labels_list.append(inst["labels"])

        # Add stuff predictions (one mask per stuff class from semantic output)
        for stuff_cls in range(30, 35):
            if stuff_cls in ignore_labels:
                continue
            stuff_mask = sem_out == stuff_cls
            if stuff_mask.any():
                pred_masks_list.append(torch.from_numpy(stuff_mask))
                pred_labels_list.append(stuff_cls)

        if pred_masks_list:
            pred_masks_t = torch.stack(pred_masks_list).bool()       # (K, N)
            pred_labels_t = torch.tensor(pred_labels_list, dtype=torch.long)  # (K,)
        else:
            pred_masks_t = torch.zeros(0, shape, dtype=torch.bool)
            pred_labels_t = torch.zeros(0, dtype=torch.long)

        # Evaluate panoptic quality if GT is available
        target_masks = det["targets"]["masks"]
        target_labels = det["targets"]["labels"]
        lengths = det["lengths"]

        # Ensure tensors (saved as torch tensors; guard against numpy fallback)
        if isinstance(target_masks, np.ndarray):
            target_masks = torch.from_numpy(target_masks)
        if isinstance(target_labels, np.ndarray):
            target_labels = torch.from_numpy(target_labels)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)

        has_gt = isinstance(target_masks, torch.Tensor) and target_masks.shape[0] > 0

        file_result = None
        if has_gt:
            preds_eval = {
                "pred_masks": [pred_masks_t],
                "pred_labels": [pred_labels_t],
            }
            targets_eval = {
                "target_masks": [target_masks.bool()],
                "target_labels": [target_labels.long()],
                "prim_lens": [lengths.float()],
            }
            if eval_mode == "strict":
                file_result = evaluator.eval_panoptic_quality_strict(preds_eval, targets_eval)
            else:
                file_result = evaluator.eval_panoptic_quality(preds_eval, targets_eval)
            total_tp     += file_result["tp_per_class"].cpu().float()
            total_fp     += file_result["fp_per_class"].cpu().float()
            total_fn     += file_result["fn_per_class"].cpu().float()
            total_tp_iou += file_result["tp_iou_score_per_class"].cpu()

        if perfile:
            if has_gt and file_result is not None and valid_all:
                pq_f, sq_f, rq_f = cal_scores(
                    file_result["tp_per_class"][valid_all].sum().float(),
                    file_result["fp_per_class"][valid_all].sum().float(),
                    file_result["fn_per_class"][valid_all].sum().float(),
                    file_result["tp_iou_score_per_class"][valid_all].sum(),
                )
            else:
                pq_f, sq_f, rq_f = 0.0, 0.0, 0.0
            perfile_results.append({
                'filename': os.path.basename(svg_path),
                'filepath': svg_path,
                'pq': pq_f,
                'rq': rq_f,
                'sq': sq_f,
            })

        coco_res.append({'filepath': det['filepath'], 'instances': instances})

        # Prepare SVG output paths
        data_name = os.path.splitext(os.path.basename(svg_path))[0]
        os.makedirs(os.path.join(out_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "svg"), exist_ok=True)

        svg_out_path = os.path.join(out_dir, "svg", f"{data_name}_predicted.svg")
        png_out_path = os.path.join(out_dir, "png", f"{data_name}_predicted.png")
        inputs.append([svg_path, sem_out.astype(np.int64), ins_out.astype(np.int64),
                       svg_out_path, png_out_path, generate_png, coords, ignore_labels])

    # Compute and print final PQ
    pq, sq, rq = cal_scores(
        total_tp[valid_all].sum(),
        total_fp[valid_all].sum(),
        total_fn[valid_all].sum(),
        total_tp_iou[valid_all].sum(),
    )
    thing_pq, thing_sq, thing_rq = cal_scores(
        total_tp[valid_thing].sum(),
        total_fp[valid_thing].sum(),
        total_fn[valid_thing].sum(),
        total_tp_iou[valid_thing].sum(),
    )
    stuff_pq, stuff_sq, stuff_rq = cal_scores(
        total_tp[valid_stuff].sum(),
        total_fp[valid_stuff].sum(),
        total_fn[valid_stuff].sum(),
        total_tp_iou[valid_stuff].sum(),
    )

    print(f"\nPQ={pq:.2f}  SQ={sq:.2f}  RQ={rq:.2f}")
    print(f"thing PQ={thing_pq:.2f}  SQ={thing_sq:.2f}  RQ={thing_rq:.2f}")
    print(f"stuff PQ={stuff_pq:.2f}  SQ={stuff_sq:.2f}  RQ={stuff_rq:.2f}")
    print(f"TP={int(total_tp[valid_all].sum())}  FP={int(total_fp[valid_all].sum())}  FN={int(total_fn[valid_all].sum())}")

    np.save(os.path.join(out_dir, "coco_res_val.npy"), coco_res)

    try:
        import mmcv
        mmcv.track_parallel_progress(process_dt_with_ids, inputs, 16)
    except ImportError:
        for inp in tqdm.tqdm(inputs, desc="Rendering SVGs"):
            process_dt_with_ids(inp)

    print(f"\nGenerated SVG files with predicted IDs in: {out_dir}/svg")
    print(f"Ignore label mode: {args.ignore_mode} ({ignore_labels})")
    print("Directory structure:")
    print(f"- SVG files saved as: {out_dir}/svg/<data_name>_predicted.svg")
    if generate_png:
        print(f"- PNG files saved as: {out_dir}/png/<data_name>_predicted.png")
    else:
        print("- PNG generation skipped (use --generate_png flag to enable)")

    # Generate per-file results markdown if requested
    if perfile and perfile_results:
        md_path = os.path.join(out_dir, "perfile_results.md")
        with open(md_path, 'w') as f:
            f.write("# Per-File PQ/RQ/SQ Results\n\n")
            f.write("| File | PQ | RQ | SQ |\n")
            f.write("|------|-----|-----|-----|\n")
            for res in perfile_results:
                f.write(f"| {res['filename']} | {res['pq']:.2f} | {res['rq']:.2f} | {res['sq']:.2f} |\n")
            f.write("\n## Aggregated Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| PQ | {pq:.2f} |\n")
            f.write(f"| RQ | {rq:.2f} |\n")
            f.write(f"| SQ | {sq:.2f} |\n")
        print(f"\nPer-file results saved to: {md_path}")
