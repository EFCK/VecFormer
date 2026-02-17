"""
VecFormer Inference Script

Standalone inference script for VecFormer model.
- Does NOT start wandb
- Does NOT save model checkpoints
- Saves inference results to output directory
- Outputs model_output.npy compatible with svg_visualize_v3.py

# made by EFCK - Custom data adaptation (ported from SymPointV2):
#   1. IGNORE_LABEL_MODES: Named presets for multi-class ignore labels (background_only, poilabs)
#   2. CLASS_NAMES: FloorPlanCAD class name list (35 foreground + background)
#   3. --ignore_mode CLI argument: Select which classes to ignore during evaluation
#   4. load_model(): Override evaluator/metrics ignore_label from CLI argument
#   5. has_ground_truth(): Detect if preprocessed data has real GT labels
#   6. compute_metrics(): Per-class PQ (TP/FP/FN), per-class IoU, mIoU/fwIoU/pACC,
#      all filtered by ignore_label
#   7. run_inference(): Conditional metric accumulation based on GT availability
#   8. save_results(): SymPointV2-style detailed log with semantic + panoptic sections,
#      per-class breakdown, total counts, GT status
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------- Default Hyperparameters ---------------------- #
DEFAULT_CHECKPOINT = "configs/cp_98"
DEFAULT_DATADIR = "datasets/FloorPlanCAD-jsons/test"
DEFAULT_OUTPUT_DIR = "./results/floorplancad_test/"
DEFAULT_DEVICE = "cuda"
DEFAULT_FP16 = False  # Disabled due to spconv compatibility issues
# --------------------------------------------------------------------- #


# Predefined ignore label modes for evaluation.
# "background_only": Only ignore background class (for original FloorPlanCAD evaluation)
# "poilabs": Ignore background + classes not present in Poilabs data
IGNORE_LABEL_MODES = {
    "background_only": [35],
    "poilabs": [35, 3, 5, 7, 8, 9, 11, 14, 15, 17, 19, 20, 21, 22, 23],
}

# FloorPlanCAD class names (0-indexed, 35 foreground classes + background at index 35).
# Matches SymPointV2's SVG_CATEGORIES (id 1-35 -> index 0-34).
CLASS_NAMES = [
    "single door",    # 0  (thing)
    "double door",    # 1  (thing)
    "sliding door",   # 2  (thing)
    "folding door",   # 3  (thing)
    "revolving door", # 4  (thing)
    "rolling door",   # 5  (thing)
    "window",         # 6  (thing)
    "bay window",     # 7  (thing)
    "blind window",   # 8  (thing)
    "opening symbol", # 9  (thing)
    "sofa",           # 10 (thing)
    "bed",            # 11 (thing)
    "chair",          # 12 (thing)
    "table",          # 13 (thing)
    "TV cabinet",     # 14 (thing)
    "Wardrobe",       # 15 (thing)
    "cabinet",        # 16 (thing)
    "gas stove",      # 17 (thing)
    "sink",           # 18 (thing)
    "refrigerator",   # 19 (thing)
    "airconditioner", # 20 (thing)
    "bath",           # 21 (thing)
    "bath tub",       # 22 (thing)
    "washing machine",# 23 (thing)
    "squat toilet",   # 24 (thing)
    "urinal",         # 25 (thing)
    "toilet",         # 26 (thing)
    "stairs",         # 27 (thing)
    "elevator",       # 28 (thing)
    "escalator",      # 29 (thing)
    "row chairs",     # 30 (stuff)
    "parking spot",   # 31 (stuff)
    "wall",           # 32 (stuff)
    "curtain wall",   # 33 (stuff)
    "railing",        # 34 (stuff)
    "background",     # 35 (background — always ignored)
]


def get_args():
    parser = argparse.ArgumentParser("VecFormer Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint directory (containing model.safetensors)",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=DEFAULT_DATADIR,
        help="Path to dataset split directory (e.g. datasets/FloorPlanCAD-jsons/test)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=DEFAULT_FP16,
        help="Use mixed precision inference",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Log total inference time and peak GPU memory usage",
    )
    parser.add_argument(
        "--ignore_mode",
        type=str,
        default="background_only",
        choices=list(IGNORE_LABEL_MODES.keys()),
        help="Ignore label mode for evaluation: "
             "'background_only' ignores only background (class 35), "
             "'poilabs' ignores background + classes not in Poilabs data",
    )
    args = parser.parse_args()
    return args


def load_model(checkpoint_path: str, device: str, ignore_label: Optional[List[int]] = None):
    """Load VecFormer model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.
        device: Device to load model on.
        ignore_label: List of class indices to ignore in evaluation.
            If provided, overrides the default config value.
    """
    from model.vecformer import VecFormer, VecFormerConfig

    # Use default model config
    config = VecFormerConfig()

    # Override ignore_label in evaluator and metrics_computer configs
    if ignore_label is not None:
        config.evaluator_config["ignore_label"] = ignore_label
        config.metrics_computer_config["ignore_label"] = ignore_label

    model = VecFormer(config)

    # Load checkpoint
    checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(checkpoint_file):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_file)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model from {checkpoint_file}")
    else:
        # Try pytorch format
        pytorch_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_file):
            state_dict = torch.load(pytorch_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {pytorch_file}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. "
                "Expected model.safetensors or pytorch_model.bin"
            )

    model = model.to(device)
    model.eval()
    model.set_inference_mode(True)
    return model, config


def load_dataset(datadir: str):
    """Load dataset for inference.

    Args:
        datadir: Path directly to the split folder (e.g. datasets/FloorPlanCAD-jsons/test)
    """
    from data.floorplancad import FloorPlanCAD

    # Eval transform args (no augmentation)
    eval_transform_args = {
        "random_vertical_flip": 0.0,
        "random_horizontal_flip": 0.0,
        "random_rotate": False,
        "random_scale": [1.0, 1.0],
        "random_translation": [0.0, 0.0],
    }

    dataset = FloorPlanCAD(
        root_dir=datadir,
        split=None,
        train_transform_args=eval_transform_args,
        eval_transform_args=eval_transform_args,
    )

    return dataset, FloorPlanCAD.collate_fn


def json_path_to_svg_path(json_path: str) -> str:
    """Convert a .json data path to the corresponding .svg path.

    The JSON and SVG datasets mirror each other's directory structure,
    with only the extension differing (.json vs .svg).
    """
    return json_path.replace(".json", ".svg")


def has_ground_truth(json_path: str) -> bool:
    """Check if a preprocessed JSON data file has ground truth labels.

    Returns True if the file has non-background semantic labels (i.e., real GT).
    Files with only background (semantic_id=35) or no labels are considered to
    have no ground truth.

    The preprocessed JSON uses field names 'semantic_ids' and 'instance_ids',
    with semantic IDs shifted to [0, 35] where 35 is background.

    Adapted from SymPointV2's has_ground_truth() function.
    """
    data = json.load(open(json_path))
    semantic_ids = data.get("semantic_ids", [])
    instance_ids = data.get("instance_ids", [])

    if len(semantic_ids) == 0:
        return False
    # If all semantic IDs are background (35), no GT
    if all(sid == 35 for sid in semantic_ids):
        return False
    # If any instance IDs are valid (>= 0), we have some GT
    if any(iid >= 0 for iid in instance_ids):
        return True
    # If we have non-background semantic IDs, we have some GT
    return any(sid < 35 for sid in semantic_ids)


def compute_metrics(
    metric_states: Optional[Dict],
    f1_states: Optional[Dict],
    config,
    ignore_label: Optional[List[int]] = None,
) -> Dict:
    """Compute final metrics from accumulated states.

    Args:
        metric_states: Accumulated PQ metric states (tp/fp/fn per class).
        f1_states: Accumulated F1 metric states (tp/pred/gt per class).
        config: VecFormerConfig with thing_class_idxs and stuff_class_idxs.
        ignore_label: List of class indices to exclude from metric computation.
            If None, defaults to empty list (all classes included).
    """
    eps = torch.finfo(torch.float32).eps
    ignore_label = ignore_label or []
    ignore_set = set(ignore_label)

    # Filter thing/stuff indices to exclude ignored classes
    thing_class_idxs = [i for i in config.thing_class_idxs if i not in ignore_set]
    stuff_class_idxs = [i for i in config.stuff_class_idxs if i not in ignore_set]
    valid_all_idxs = thing_class_idxs + stuff_class_idxs

    metrics = {}

    # Panoptic quality metrics
    if metric_states:
        tp = metric_states["tp_per_class"].float()
        fp = metric_states["fp_per_class"].float()
        fn = metric_states["fn_per_class"].float()
        tp_iou = metric_states["tp_iou_score_per_class"].float()

        def calc_pq_metrics(tp, fp, fn, tp_iou):
            rq = tp / (tp + 0.5 * fn + 0.5 * fp + eps)
            sq = tp_iou / (tp + eps)
            pq = rq * sq
            return pq, sq, rq

        # Overall metrics (only valid classes)
        pq, sq, rq = calc_pq_metrics(
            tp[valid_all_idxs].sum(),
            fp[valid_all_idxs].sum(),
            fn[valid_all_idxs].sum(),
            tp_iou[valid_all_idxs].sum(),
        )
        metrics["PQ"] = pq.item() * 100
        metrics["SQ"] = sq.item() * 100
        metrics["RQ"] = rq.item() * 100

        # Thing metrics (only valid thing classes)
        thing_pq, thing_sq, thing_rq = calc_pq_metrics(
            tp[thing_class_idxs].sum(),
            fp[thing_class_idxs].sum(),
            fn[thing_class_idxs].sum(),
            tp_iou[thing_class_idxs].sum(),
        )
        metrics["thing_PQ"] = thing_pq.item() * 100
        metrics["thing_SQ"] = thing_sq.item() * 100
        metrics["thing_RQ"] = thing_rq.item() * 100

        # Stuff metrics (only valid stuff classes)
        stuff_pq, stuff_sq, stuff_rq = calc_pq_metrics(
            tp[stuff_class_idxs].sum(),
            fp[stuff_class_idxs].sum(),
            fn[stuff_class_idxs].sum(),
            tp_iou[stuff_class_idxs].sum(),
        )
        metrics["stuff_PQ"] = stuff_pq.item() * 100
        metrics["stuff_SQ"] = stuff_sq.item() * 100
        metrics["stuff_RQ"] = stuff_rq.item() * 100

        # Per-class PQ breakdown
        per_class_pq = {}
        for i in range(len(tp)):
            rq_i = tp[i] / (tp[i] + 0.5 * fn[i] + 0.5 * fp[i] + eps)
            sq_i = tp_iou[i] / (tp[i] + eps)
            per_class_pq[i] = {
                "PQ": (rq_i * sq_i).item() * 100,
                "SQ": sq_i.item() * 100,
                "RQ": rq_i.item() * 100,
                "TP": int(tp[i].item()),
                "FP": int(fp[i].item()),
                "FN": int(fn[i].item()),
                "ignored": i in ignore_set,
            }
        metrics["per_class_PQ"] = per_class_pq

        # Total TP/FP/FN (valid classes only)
        metrics["total_TP"] = int(tp[valid_all_idxs].sum().item())
        metrics["total_FP"] = int(fp[valid_all_idxs].sum().item())
        metrics["total_FN"] = int(fn[valid_all_idxs].sum().item())

    # F1 / semantic segmentation metrics (primitive-length-weighted)
    if f1_states:
        # f1_states are shaped (num_classes+1,) — index by valid class idxs
        tp_per = f1_states["tp_per_class"].float()
        pred_per = f1_states["pred_per_class"].float()
        gt_per = f1_states["gt_per_class"].float()
        tp = tp_per[valid_all_idxs].sum()
        pred = pred_per[valid_all_idxs].sum()
        gt = gt_per[valid_all_idxs].sum()
        precision = tp / (pred + eps)
        recall = tp / (gt + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        metrics["F1"] = f1.item()

        # Weighted (primitive-length-weighted) tensors for IoU-based metrics
        w_tp_per = f1_states["w_tp_per_class"].float()
        w_pred_per = f1_states["w_pred_per_class"].float()
        w_gt_per = f1_states["w_gt_per_class"].float()

        # Per-class IoU = w_tp / (w_pred + w_gt - w_tp)
        num_sem_classes = len(w_tp_per) - 1  # last index is background
        per_class_iou = {}
        for i in range(num_sem_classes):
            union = w_pred_per[i] + w_gt_per[i] - w_tp_per[i]
            per_class_iou[i] = {
                "IoU": (w_tp_per[i] / (union + eps)).item() * 100,
                "ignored": i in ignore_set,
            }
        metrics["per_class_IoU"] = per_class_iou

        # mIoU over valid classes
        valid_ious = [per_class_iou[i]["IoU"] for i in valid_all_idxs]
        metrics["mIoU"] = float(np.mean(valid_ious)) if valid_ious else 0.0

        # fwIoU: frequency-weighted IoU (weight = gt primitives / total gt primitives)
        total_gt_w = w_gt_per[valid_all_idxs].sum()
        fw_iou = 0.0
        if total_gt_w > 0:
            for i in valid_all_idxs:
                union = w_pred_per[i] + w_gt_per[i] - w_tp_per[i]
                iou_i = w_tp_per[i] / (union + eps)
                fw_iou += (w_gt_per[i] / total_gt_w * iou_i).item()
        metrics["fwIoU"] = fw_iou * 100

        # Pixel accuracy over valid classes
        metrics["pACC"] = (
            w_tp_per[valid_all_idxs].sum() / (w_gt_per[valid_all_idxs].sum() + eps)
        ).item() * 100

        # Aggregate wF1
        w_tp = w_tp_per[valid_all_idxs].sum()
        w_pred = w_pred_per[valid_all_idxs].sum()
        w_gt = w_gt_per[valid_all_idxs].sum()
        w_precision = w_tp / (w_pred + eps)
        w_recall = w_tp / (w_gt + eps)
        w_f1 = 2 * w_precision * w_recall / (w_precision + w_recall + eps)
        metrics["wF1"] = w_f1.item()

    return metrics


def build_save_dict(
    data_path: str,
    inst_masks: torch.Tensor,
    inst_labels: torch.Tensor,
    inst_scores: torch.Tensor,
    sem_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    prim_lens: torch.Tensor,
) -> Dict:
    """Build a SymPointV2-compatible save dict for one sample.

    The output format matches what svg_visualize_v3.py expects:
        - filepath: SVG file path
        - sem: semantic scores array [N, C] (softmax probabilities)
        - ins: list of instance dicts with {masks, labels, scores}
        - targets: dict with {labels, masks} ground truth
        - lengths: primitive lengths tensor

    Args:
        data_path: JSON data path from the dataset
        inst_masks: (K, N) boolean instance masks from dict_pred_inst_segs
        inst_labels: (K,) instance labels
        inst_scores: (K,) instance confidence scores
        sem_logits: (N, C) raw semantic logits before argmax
        target_labels: (M,) ground truth panoptic labels
        target_masks: (M, N) ground truth panoptic masks
        prim_lens: (N,) primitive lengths
    """
    # Convert JSON path to SVG path
    filepath = json_path_to_svg_path(data_path)

    # Semantic scores: softmax of raw logits -> [N, C] numpy array
    sem_scores = sem_logits.softmax(dim=-1).cpu().numpy()

    # Instance predictions: list of dicts matching SymPointV2 format
    ins_list = []
    inst_masks_cpu = inst_masks.cpu().numpy()
    inst_labels_cpu = inst_labels.cpu().numpy()
    inst_scores_cpu = inst_scores.cpu().numpy()
    for k in range(len(inst_labels_cpu)):
        ins_list.append({
            "masks": inst_masks_cpu[k].astype(bool),  # (N,) boolean
            "labels": int(inst_labels_cpu[k]),
            "scores": float(inst_scores_cpu[k]),
        })

    # Ground truth targets
    targets = {
        "labels": target_labels.cpu(),
        "masks": target_masks.cpu(),
    }

    # Primitive lengths
    lengths = prim_lens.cpu()

    return {
        "filepath": filepath,
        "sem": sem_scores,
        "ins": ins_list,
        "targets": targets,
        "lengths": lengths,
    }


def run_inference(
    model,
    dataset,
    collate_fn,
    config,
    device: str,
    fp16: bool = False,
    output_dir: str = "./results/floorplancad_test/",
    profile: bool = False,
    ignore_label: Optional[List[int]] = None,
    has_gt_data: bool = True,
):
    """Run inference on dataset and return results with SymPointV2-compatible save dicts.

    Args:
        has_gt_data: Whether the dataset has ground truth labels. If False,
            metric accumulation is skipped and metrics dict will be empty.
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for variable length sequences
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Accumulated metrics
    accumulated_metric_states = None
    accumulated_f1_states = None

    # Results storage
    save_dicts = []
    inference_times = []
    skipped_files = []
    primitive_counts = []

    logger.info(f"Running inference on {len(dataset)} samples...")

    if profile:
        torch.cuda.reset_peak_memory_stats()
        # Initialize pynvml for physical VRAM tracking
        import pynvml
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        per_sample_allocated = []  # per-sample peak allocated VRAM in bytes
        per_sample_reserved = []   # per-sample peak reserved VRAM in bytes
        per_sample_physical = []   # per-sample physical VRAM snapshot in bytes

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Inference")):
            try:
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value

                torch.cuda.empty_cache()

                # Track primitive count (real primitives, excluding padding)
                primitive_counts.append(int(torch.count_nonzero(batch_device["prim_lengths"]).item()))

                # Run inference
                with torch.amp.autocast("cuda", enabled=fp16):
                    if profile:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    t1 = time.monotonic()
                    outputs = model(**batch_device)
                    if profile:
                        torch.cuda.synchronize()
                    t2 = time.monotonic()
                    inference_times.append(t2 - t1)

                    # Track VRAM usage
                    if profile:
                        per_sample_allocated.append(torch.cuda.max_memory_allocated())
                        per_sample_reserved.append(torch.cuda.max_memory_reserved())
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                        per_sample_physical.append(mem_info.used)

                # Accumulate metric states (only when GT is available)
                if has_gt_data and outputs.metric_states is not None:
                    if accumulated_metric_states is None:
                        accumulated_metric_states = {
                            k: v.clone() for k, v in outputs.metric_states.items()
                        }
                    else:
                        for k, v in outputs.metric_states.items():
                            accumulated_metric_states[k] += v

                if has_gt_data and outputs.f1_states is not None:
                    if accumulated_f1_states is None:
                        accumulated_f1_states = {
                            k: v.clone() for k, v in outputs.f1_states.items()
                        }
                    else:
                        for k, v in outputs.f1_states.items():
                            accumulated_f1_states[k] += v

                # Build SymPointV2-compatible save dicts
                if (outputs.dict_pred_inst_segs is not None
                        and outputs.raw_sem_logits is not None
                        and outputs.targets is not None):
                    data_paths = batch_device.get("data_paths", [f"sample_{batch_idx}"])
                    for i, data_path in enumerate(data_paths):
                        save_dict = build_save_dict(
                            data_path=data_path,
                            inst_masks=outputs.dict_pred_inst_segs["list_pred_masks"][i],
                            inst_labels=outputs.dict_pred_inst_segs["list_pred_labels"][i],
                            inst_scores=outputs.dict_pred_inst_segs["list_pred_scores"][i],
                            sem_logits=outputs.raw_sem_logits[i],
                            target_labels=outputs.targets["target_labels"][i],
                            target_masks=outputs.targets["target_masks"][i],
                            prim_lens=outputs.targets["prim_lens"][i],
                        )
                        save_dicts.append(save_dict)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    data_paths = batch.get("data_paths", [f"sample_{batch_idx}"])
                    skipped_files.extend(data_paths)
                    logger.warning(f"[OOM] Skipped batch {batch_idx}")
                    continue
                else:
                    raise e

    # Compute final metrics
    metrics = compute_metrics(
        accumulated_metric_states, accumulated_f1_states, config,
        ignore_label=ignore_label,
    )

    # Timing statistics
    avg_time = np.mean(inference_times) if inference_times else 0
    total_time = np.sum(inference_times) if inference_times else 0

    # Build timing dict
    timing = {
        "total_time": total_time,
        "avg_time_per_sample": avg_time,
        "num_samples": len(inference_times),
        "primitives_min": int(np.min(primitive_counts)) if primitive_counts else 0,
        "primitives_mean": round(float(np.mean(primitive_counts)), 1) if primitive_counts else 0,
        "primitives_max": int(np.max(primitive_counts)) if primitive_counts else 0,
        "primitives_std": round(float(np.std(primitive_counts)), 1) if primitive_counts else 0,
    }

    if profile and inference_times:
        min_time = float(np.min(inference_times))
        max_time = float(np.max(inference_times))
        allocated_mb = np.array(per_sample_allocated) / (1024 ** 2)
        reserved_mb = np.array(per_sample_reserved) / (1024 ** 2)
        physical_mb = np.array(per_sample_physical) / (1024 ** 2)
        pynvml.nvmlShutdown()

        # Add profile stats to timing dict
        timing["min_time_per_sample"] = min_time
        timing["max_time_per_sample"] = max_time
        timing["std_time_per_sample"] = float(np.std(inference_times))
        timing["vram_allocated_min_mb"] = round(float(np.min(allocated_mb)), 1)
        timing["vram_allocated_mean_mb"] = round(float(np.mean(allocated_mb)), 1)
        timing["vram_allocated_max_mb"] = round(float(np.max(allocated_mb)), 1)
        timing["vram_allocated_std_mb"] = round(float(np.std(allocated_mb)), 1)
        timing["vram_reserved_min_mb"] = round(float(np.min(reserved_mb)), 1)
        timing["vram_reserved_mean_mb"] = round(float(np.mean(reserved_mb)), 1)
        timing["vram_reserved_max_mb"] = round(float(np.max(reserved_mb)), 1)
        timing["vram_reserved_std_mb"] = round(float(np.std(reserved_mb)), 1)
        timing["vram_physical_min_mb"] = round(float(np.min(physical_mb)), 1)
        timing["vram_physical_mean_mb"] = round(float(np.mean(physical_mb)), 1)
        timing["vram_physical_max_mb"] = round(float(np.max(physical_mb)), 1)
        timing["vram_physical_std_mb"] = round(float(np.std(physical_mb)), 1)

        # Console output
        logger.info("=" * 60)
        logger.info("PROFILE RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total time          : {total_time:.2f}s")
        logger.info(f"  Mean time           : {avg_time:.4f}s")
        logger.info(f"  Min time            : {min_time:.4f}s")
        logger.info(f"  Max time            : {max_time:.4f}s")
        logger.info(f"  Std time            : {float(np.std(inference_times)):.4f}s")
        logger.info(f"  VRAM allocated (min/mean/max/std): {np.min(allocated_mb):.1f} / {np.mean(allocated_mb):.1f} / {np.max(allocated_mb):.1f} / {np.std(allocated_mb):.1f} MB")
        logger.info(f"  VRAM reserved  (min/mean/max/std): {np.min(reserved_mb):.1f} / {np.mean(reserved_mb):.1f} / {np.max(reserved_mb):.1f} / {np.std(reserved_mb):.1f} MB")
        logger.info(f"  VRAM physical  (min/mean/max/std): {np.min(physical_mb):.1f} / {np.mean(physical_mb):.1f} / {np.max(physical_mb):.1f} / {np.std(physical_mb):.1f} MB")
        logger.info(f"  Primitives (min/mean/max/std): {timing['primitives_min']} / {timing['primitives_mean']} / {timing['primitives_max']} / {timing['primitives_std']}")

    return {
        "metrics": metrics,
        "save_dicts": save_dicts,
        "timing": timing,
        "skipped_files": skipped_files,
        "has_gt_data": has_gt_data,
    }


def save_results(results: Dict, output_dir: str, args):
    """Save inference results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    has_gt = results.get("has_gt_data", True)
    metrics = results["metrics"]
    timing = results["timing"]
    skipped = results["skipped_files"]
    ignore_label = IGNORE_LABEL_MODES[args.ignore_mode]
    predicted = timing["num_samples"]
    evaluated = predicted if has_gt else 0

    # ------------------------------------------------------------------ #
    # Log file                                                             #
    # ------------------------------------------------------------------ #
    log_file = os.path.join(output_dir, f"inference_results_{timestamp_file}.log")
    with open(log_file, "w") as f:
        SEP = "=" * 60
        sep = "-" * 60

        # Header
        f.write(SEP + "\n")
        f.write("INFERENCE RESULTS\n")
        f.write(SEP + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.datadir}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"FP16: {args.fp16}\n")
        f.write(f"Ignore label mode: {args.ignore_mode} ({ignore_label})\n")
        f.write(f"Total files: {predicted}\n")
        f.write(f"Predicted files: {predicted}\n")
        f.write(f"Evaluated files (with GT): {evaluated}\n")
        f.write(f"Skipped files (OOM): {len(skipped)}\n")
        if not has_gt:
            f.write("Ground truth: not available — metrics not computed\n")
        f.write("\n")

        # Timing
        f.write(sep + "\n")
        f.write("TIMING\n")
        f.write(sep + "\n")
        f.write(f"Total inference time: {timing['total_time']:.2f}s\n")
        f.write(f"Average time per file: {timing['avg_time_per_sample']:.4f}s\n")
        f.write(
            f"Primitives (min/mean/max/std): "
            f"{timing['primitives_min']} / {timing['primitives_mean']} / "
            f"{timing['primitives_max']} / {timing['primitives_std']}\n"
        )
        if "min_time_per_sample" in timing:
            f.write(f"Min time per file: {timing['min_time_per_sample']:.4f}s\n")
            f.write(f"Max time per file: {timing['max_time_per_sample']:.4f}s\n")
            f.write(f"Std time per file: {timing['std_time_per_sample']:.4f}s\n")
            f.write(
                f"VRAM allocated (min/mean/max/std): "
                f"{timing['vram_allocated_min_mb']:.1f} / {timing['vram_allocated_mean_mb']:.1f} / "
                f"{timing['vram_allocated_max_mb']:.1f} / {timing['vram_allocated_std_mb']:.1f} MB\n"
            )
            f.write(
                f"VRAM reserved  (min/mean/max/std): "
                f"{timing['vram_reserved_min_mb']:.1f} / {timing['vram_reserved_mean_mb']:.1f} / "
                f"{timing['vram_reserved_max_mb']:.1f} / {timing['vram_reserved_std_mb']:.1f} MB\n"
            )
            f.write(
                f"VRAM physical  (min/mean/max/std): "
                f"{timing['vram_physical_min_mb']:.1f} / {timing['vram_physical_mean_mb']:.1f} / "
                f"{timing['vram_physical_max_mb']:.1f} / {timing['vram_physical_std_mb']:.1f} MB\n"
            )
        f.write("\n")

        if not metrics:
            f.write(sep + "\n")
            f.write("METRICS\n")
            f.write(sep + "\n")
            f.write("\nNo evaluation metrics — ground truth labels not available.\n\n")
        else:
            # Semantic Segmentation section
            if "mIoU" in metrics:
                f.write(sep + "\n")
                f.write("SEMANTIC SEGMENTATION\n")
                f.write(sep + "\n")
                f.write(f"mIoU:  {metrics['mIoU']:.3f}\n")
                f.write(f"fwIoU: {metrics['fwIoU']:.3f}\n")
                f.write(f"pACC:  {metrics['pACC']:.3f}\n")
                f.write("\nPer-class IoU:\n")
                per_iou = metrics["per_class_IoU"]
                for i, name in enumerate(CLASS_NAMES[:-1]):  # skip background
                    if i not in per_iou or per_iou[i]["ignored"]:
                        continue
                    f.write(f"  {name:<22}: {per_iou[i]['IoU']:>7.3f}\n")
                f.write("\n")

            # Panoptic Segmentation section
            if "PQ" in metrics:
                f.write(sep + "\n")
                f.write("PANOPTIC SEGMENTATION\n")
                f.write(sep + "\n")
                f.write(
                    f"PQ: {metrics['PQ']:.3f}  |  RQ: {metrics['RQ']:.3f}  |  SQ: {metrics['SQ']:.3f}\n\n"
                )
                f.write("Thing classes:\n")
                f.write(
                    f"  PQ: {metrics['thing_PQ']:.3f}  |  RQ: {metrics['thing_RQ']:.3f}  |  SQ: {metrics['thing_SQ']:.3f}\n\n"
                )
                f.write("Stuff classes:\n")
                f.write(
                    f"  PQ: {metrics['stuff_PQ']:.3f}  |  RQ: {metrics['stuff_RQ']:.3f}  |  SQ: {metrics['stuff_SQ']:.3f}\n\n"
                )
                f.write("Per-class PQ:\n")
                per_pq = metrics["per_class_PQ"]
                for i, name in enumerate(CLASS_NAMES[:-1]):  # skip background
                    if i not in per_pq or per_pq[i]["ignored"]:
                        continue
                    c = per_pq[i]
                    f.write(
                        f"  {name:<22}: PQ={c['PQ']:>7.3f}  RQ={c['RQ']:>7.3f}  SQ={c['SQ']:>7.3f}"
                        f"  (TP={c['TP']}, FP={c['FP']}, FN={c['FN']})\n"
                    )
                f.write("\n")

            # Total counts
            if "total_TP" in metrics:
                f.write(sep + "\n")
                f.write("TOTAL COUNTS\n")
                f.write(sep + "\n")
                f.write(f"TP: {metrics['total_TP']}\n")
                f.write(f"FP: {metrics['total_FP']}\n")
                f.write(f"FN: {metrics['total_FN']}\n\n")

        # Skipped files
        if skipped:
            f.write(sep + "\n")
            f.write("SKIPPED FILES (OOM)\n")
            f.write(sep + "\n")
            for s in skipped:
                f.write(f"  {s}\n")

    logger.info(f"Saved results log to {log_file}")

    # ------------------------------------------------------------------ #
    # JSON metrics file                                                    #
    # ------------------------------------------------------------------ #
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp_file}.json")
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "metrics": results["metrics"],
                "timing": results["timing"],
                "skipped_files": skipped,
                "has_ground_truth": has_gt,
                "ignore_mode": args.ignore_mode,
                "ignore_label": ignore_label,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics JSON to {metrics_file}")

    # ------------------------------------------------------------------ #
    # model_output.npy (SymPointV2-compatible)                            #
    # ------------------------------------------------------------------ #
    if results["save_dicts"]:
        npy_file = os.path.join(output_dir, "model_output.npy")
        np.save(npy_file, results["save_dicts"])
        logger.info(f"Saved {len(results['save_dicts'])} predictions to {npy_file}")


def main():
    args = get_args()

    # Resolve ignore labels from --ignore_mode
    ignore_label = IGNORE_LABEL_MODES[args.ignore_mode]

    logger.info("=" * 60)
    logger.info("VecFormer Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.datadir}")
    logger.info(f"Output Dir: {args.out}")
    logger.info(f"Device: {args.device}")
    logger.info(f"FP16: {args.fp16}")
    logger.info(f"Ignore mode: {args.ignore_mode} ({ignore_label})")

    # Load model
    logger.info("\nLoading model...")
    model, config = load_model(args.checkpoint, args.device, ignore_label=ignore_label)

    # Load dataset
    logger.info("Loading dataset...")
    dataset, collate_fn = load_dataset(args.datadir)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Check if dataset has ground truth labels
    data_dir = dataset.data_dir
    data_paths = dataset.data_paths
    has_gt_data = any(
        has_ground_truth(os.path.join(data_dir, p)) for p in data_paths
    )
    if has_gt_data:
        logger.info("Ground truth labels detected - evaluation metrics will be computed")
    else:
        logger.info("No ground truth labels found - skipping evaluation metrics")

    # Run inference
    results = run_inference(
        model=model,
        dataset=dataset,
        collate_fn=collate_fn,
        config=config,
        device=args.device,
        fp16=args.fp16,
        output_dir=args.out,
        profile=args.profile,
        ignore_label=ignore_label,
        has_gt_data=has_gt_data,
    )

    # Print summary to console
    metrics = results["metrics"]
    timing = results["timing"]
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    if not metrics:
        logger.info("No evaluation metrics (no ground truth)")
    else:
        if "mIoU" in metrics:
            logger.info(f"mIoU: {metrics['mIoU']:.3f}  |  fwIoU: {metrics['fwIoU']:.3f}  |  pACC: {metrics['pACC']:.3f}")
        if "PQ" in metrics:
            logger.info(f"PQ: {metrics['PQ']:.3f}  |  RQ: {metrics['RQ']:.3f}  |  SQ: {metrics['SQ']:.3f}")
            logger.info(f"  Thing  PQ: {metrics['thing_PQ']:.3f}  |  RQ: {metrics['thing_RQ']:.3f}  |  SQ: {metrics['thing_SQ']:.3f}")
            logger.info(f"  Stuff  PQ: {metrics['stuff_PQ']:.3f}  |  RQ: {metrics['stuff_RQ']:.3f}  |  SQ: {metrics['stuff_SQ']:.3f}")
        if "total_TP" in metrics:
            logger.info(f"  TP: {metrics['total_TP']}  FP: {metrics['total_FP']}  FN: {metrics['total_FN']}")

    logger.info(f"Inference time: {timing['total_time']:.2f}s ({timing['avg_time_per_sample']:.4f}s/file)")
    logger.info(f"Predictions saved: {len(results['save_dicts'])}")

    if results["skipped_files"]:
        logger.warning(f"Skipped {len(results['skipped_files'])} files due to OOM")

    # Save results
    save_results(results, args.out, args)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
