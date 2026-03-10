import json, os, glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from svgpathtools import parse_path
import re, math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

from model.vecformer.evaluator.evaluator import Evaluator, EvaluatorConfig


# FloorPlanCAD 36-class color table (0-indexed; index == VecFormer label).
# Thing classes: indices 0-29 (ids 1-30). Stuff classes: indices 30-34 (ids 31-35).
# Background: index 35 (id 36).
SVG_CATEGORIES = [
    {"color": [224,  62, 155], "isthing": 1, "id":  1, "name": "single door"},
    {"color": [119,  11,  32], "isthing": 1, "id":  2, "name": "double door"},
    {"color": [  0,   0, 142], "isthing": 1, "id":  3, "name": "sliding door"},
    {"color": [  0,   0, 230], "isthing": 1, "id":  4, "name": "folding door"},
    {"color": [220,  20,  60], "isthing": 1, "id":  5, "name": "revolving door"},
    {"color": [255,   0,   0], "isthing": 1, "id":  6, "name": "rolling door"},
    {"color": [  0,  60, 100], "isthing": 1, "id":  7, "name": "window"},
    {"color": [  0,  80, 100], "isthing": 1, "id":  8, "name": "bay window"},
    {"color": [  0,   0, 192], "isthing": 1, "id":  9, "name": "blind window"},
    {"color": [128,  64, 255], "isthing": 1, "id": 10, "name": "opening symbol"},
    {"color": [  0, 128, 192], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [  0,  64,  64], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [  0, 192,  64], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [128, 128,   0], "isthing": 1, "id": 14, "name": "table"},
    {"color": [  0, 128, 128], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [128,   0,  64], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [192, 128,   0], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [192,   0,   0], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [  0,   0, 128], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [  0, 192, 128], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [192,   0, 128], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [  0, 128,  64], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [ 64,   0, 192], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [192,  64,   0], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [128, 192,   0], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [ 64, 128,   0], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [  0,  64, 128], "isthing": 1, "id": 27, "name": "toilet"},
    {"color": [192, 192,   0], "isthing": 1, "id": 28, "name": "stairs"},
    {"color": [ 64, 192,   0], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [  0, 192, 192], "isthing": 1, "id": 30, "name": "escalator"},
    {"color": [ 64,  64, 192], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [192,  64,  64], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [192, 128, 128], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [128, 192, 128], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [128, 128, 192], "isthing": 0, "id": 35, "name": "railing"},
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


def svg_reader(svg_path):
    svg_list = list()
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print("Read{} failed!".format(svg_path))
        return svg_list
    root = tree.getroot()
    for elem in root.iter():
        line = elem.attrib
        line['tag'] = elem.tag
        svg_list.append(line)
    return svg_list



def svg_writer(svg_list, svg_path):
    root = None
    current_parent = None

    for idx, line in enumerate(svg_list):
        tag = line["tag"]
        line.pop("tag")

        if idx == 0:
            root = ET.Element(tag)
            root.attrib = line
            current_parent = root
        else:
            if "}g" in tag:
                group = ET.SubElement(root, tag)
                group.attrib = line
                current_parent = group
            else:
                # Support both grouped and ungrouped SVG formats
                if current_parent is None:
                    current_parent = root
                node = ET.SubElement(current_parent, tag)
                node.attrib = line

    from xml.dom import minidom
    reparsed = minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t")
    f = open(svg_path,'w',encoding='utf-8')
    f.write(reparsed)
    f.close()

def visualSVG(parsing_list,labels,out_path,cvt_color=False):


    ind = 0
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text'],tag+" is error!!"
        if tag in ["path","circle",]:
            label = int(line["semanticIds"]) if "semanticIds" in line.keys() else -1
            label = labels[ind]
            color = SVG_CATEGORIES[label]["color"]
            line["stroke"] = "rgb({:d},{:d},{:d})".format(color[0],color[1],color[2])
            line["fill"] = "none"
            line["stroke-width"] = "0.2"
            ind += 1

        if tag == "svg":
            viewBox = line["viewBox"]
            viewBox = viewBox.split(" ")
            line["viewBox"] = " ".join(viewBox)
            if cvt_color:
                line["style"] = "background-color: #255255255;"


    svg_writer(parsing_list, out_path)
    return out_path

def visualSVG_with_ids(parsing_list, sem_labels, ins_labels, out_path, ignore_labels=None, cvt_color=False):
    """
    Visualize SVG with model predictions for both semantic and instance IDs.

    Args:
        ignore_labels: List of label indices to filter out (map to background).
                       If None, no filtering is applied.
    """
    if ignore_labels is None:
        ignore_labels = []

    # Get valid primitives (length >= 1e-10) like in parsing
    # We need to compute lengths to know which primitives were actually fed to the model
    # Model drops primitives with length < 1e-10
    
    valid_parsing_list = []
    ind = 0
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text'], tag+" is error!!"

        if tag in ["path", "circle", "ellipse"]:
            # Check if this primitive is valid (length >= 1e-10)
            is_valid = True
            if "d" in line:
                try:
                    path_repre = parse_path(line['d'])
                    if path_repre.length() < 1e-10:
                        is_valid = False
                except Exception:
                    pass
            elif "r" in line:  # circle
                try:
                    if 2 * math.pi * float(line['r']) < 1e-10:
                        is_valid = False
                except (ValueError, KeyError):
                    pass
            if not is_valid:
                continue

            # Get model predictions
            try:
                sem_label = sem_labels[ind]
            except IndexError:
                print(f"\nERROR: out of bounds in {out_path}")
                print(f"tag: {tag}")
                print(f"ind: {ind}, len(sem_labels): {len(sem_labels)}, count svg tags: {len(parsing_list)}")
                # count how many drawable primitives
                prims = sum(1 for p in parsing_list if p["tag"].split("svg}")[-1] in ["path", "circle", "ellipse"])
                print(f"Drawable primitives in parsed SVG: {prims}")
                print(f"sem_labels shape: {sem_labels.shape}")
                raise
            ins_label = ins_labels[ind]

            # Update semantic and instance IDs in SVG attributes
            if sem_label in ignore_labels:
                sem_label = 35 # Map to background
                if "semanticId" in line:
                    line.pop("semanticId")
                if "instanceId" in line:
                    line.pop("instanceId")
            else:
                line["semanticId"] = str(int(sem_label) + 1)  # +1 because SVG uses 1-based indexing
                if sem_label in [30, 31, 32, 33, 34]:
                    line["instanceId"] = "-1"
                else:
                    line["instanceId"] = str(int(ins_label))

            # Set color based on semantic prediction
            color = SVG_CATEGORIES[sem_label]["color"]
            line["stroke"] = "rgb({:d},{:d},{:d})".format(color[0], color[1], color[2])
            line["fill"] = "none"
            line["stroke-width"] = "0.2"
            ind += 1

        if tag == "svg":
            viewBox = line["viewBox"]
            viewBox = viewBox.split(" ")
            line["viewBox"] = " ".join(viewBox)
            if cvt_color:
                line["style"] = "background-color: #255255255;"

        valid_parsing_list.append(line)

    svg_writer(valid_parsing_list, out_path)
    return out_path


def process_dt(input):
    parsing_list, labels, out_path, generate_png = input

    visualSVG(parsing_list, labels, out_path)
    if generate_png:
        svg2png(out_path)

def process_dt_with_ids(input):
    parsing_list, sem_labels, ins_labels, out_path, png_out_path, generate_png, coords, ignore_labels = input

    visualSVG_with_ids(parsing_list, sem_labels, ins_labels, out_path, ignore_labels=ignore_labels)

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



def get_path(svg_lists):
    args, widths, gids, lengths, types = [], [], [], [], []
    COMMANDS = ['Line', 'Arc','circle', 'ellipse']
    for line in svg_lists:

        if "d" in line.keys():
            path_repre = parse_path(line['d'])
            # Skip zero-length paths — model preprocessor drops these (prim_length < 1e-10)
            try:
                path_length = path_repre.length()
            except (RuntimeError, ValueError):
                path_length = 0.0
            if path_length < 1e-10:
                continue

            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            inds = [0, 1/3, 2/3, 1.0]
            arg = []
            try:
                for ind in inds:
                    point = path_repre.point(ind)
                    arg.extend([point.real, point.imag])
            except (RuntimeError, ValueError):
                try:
                    start_point = path_repre[0].start
                    for _ in inds:
                        arg.extend([start_point.real, start_point.imag])
                except (AttributeError, IndexError):
                    for _ in inds:
                        arg.extend([0.0, 0.0])

            args.append(arg)
            lengths.append(path_length)
            path_type = path_repre[0].__class__.__name__
            types.append(COMMANDS.index(path_type))
        elif "r" in line.keys():
            r = float(line['r'])
            circle_len = 2 * math.pi * r
            # Skip zero-radius circles — model preprocessor drops these
            if circle_len < 1e-10:
                continue
            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            cx = float(line['cx'])
            cy = float(line['cy'])
            arg = []
            thetas = [0, math.pi/2, math.pi, 3 * math.pi/2]
            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x, y])
            args.append(arg)
            lengths.append(circle_len)
            types.append(COMMANDS.index("circle"))
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

    print(f"Ignore label mode: {args.ignore_mode} ({ignore_labels})")

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

        parsing_list = svg_reader(svg_path)
        widths, gids, path_args, lengths_arr, types = get_path(parsing_list)
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
        inputs.append([parsing_list, sem_out.astype(np.int64), ins_out.astype(np.int64),
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
