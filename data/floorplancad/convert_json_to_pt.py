# made by efck (2026-03-10): one-time conversion script to pre-tensorize FloorPlanCAD JSON files,
#   eliminating json.load() + to_tensor() overhead in FloorPlanCAD.__getitem__ during training.

"""Convert preprocessed JSON floor plan files to pre-tensorized .pt files.

Eliminates JSON parsing + list→tensor conversion overhead in FloorPlanCAD.__getitem__
by pre-running to_tensor() once and saving the result. Training then uses torch.load()
instead of json.load() + SVGData construction + to_tensor().

The output directory mirrors the input directory structure exactly, with .json → .pt.

Usage:
    python data/floorplancad/convert_json_to_pt.py \
        --input_dir /path/to/line_json/FloorPlanCAD \
        --output_dir /path/to/line_pt/FloorPlanCAD \
        --num_workers 16
"""

import os
import json
import argparse
from pathlib import Path

import torch

from utils.svg_util import scan_dir
from utils.parallel_mapper import parallel_map
from data.floorplancad.dataclass_define import SVGData
from data.floorplancad.transform_utils import to_tensor


def convert_one(rel_path: str, input_dir: str, output_dir: str) -> None:
    pt_path = Path(output_dir) / Path(rel_path).with_suffix(".pt")
    if pt_path.exists():
        return
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(input_dir, rel_path)
    with open(json_path, "r") as f:
        json_data = json.load(f)
    svg_data = SVGData(**json_data)
    svg_data_tensor = to_tensor(svg_data)
    torch.save(svg_data_tensor.__dict__, str(pt_path))


def main():
    parser = argparse.ArgumentParser(
        description="Convert FloorPlanCAD JSON files to pre-tensorized .pt files"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root directory containing .json files (e.g. line_json/FloorPlanCAD)")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory to write .pt files (e.g. line_pt/FloorPlanCAD)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel worker processes (default: 16)")
    args = parser.parse_args()

    rel_paths = scan_dir(args.input_dir, suffix=".json")
    print(f"Found {len(rel_paths)} JSON files in {args.input_dir}")

    parallel_map(
        convert_one,
        rel_paths,
        [args.input_dir] * len(rel_paths),
        [args.output_dir] * len(rel_paths),
        max_workers=args.num_workers,
        description="Converting JSON → .pt",
        use_progress_bar=True,
    )
    print(f"Done. .pt files written to {args.output_dir}")


if __name__ == "__main__":
    main()
