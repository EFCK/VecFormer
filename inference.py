"""
VecFormer Inference Script

Standalone inference script for VecFormer model.
- Does NOT start wandb
- Does NOT save model checkpoints
- Saves inference results to output directory
- Outputs model_output.npy compatible with svg_visualize_v3.py
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict

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
    args = parser.parse_args()
    return args


def load_model(checkpoint_path: str, device: str):
    """Load VecFormer model from checkpoint"""
    from model.vecformer import VecFormer, VecFormerConfig

    # Use default model config
    config = VecFormerConfig()
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


def compute_metrics(metric_states: Dict, f1_states: Dict, config) -> Dict:
    """Compute final metrics from accumulated states"""
    eps = torch.finfo(torch.float32).eps
    thing_class_idxs = config.thing_class_idxs
    stuff_class_idxs = config.stuff_class_idxs

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

        # Overall metrics
        pq, sq, rq = calc_pq_metrics(tp.sum(), fp.sum(), fn.sum(), tp_iou.sum())
        metrics["PQ"] = pq.item() * 100
        metrics["SQ"] = sq.item() * 100
        metrics["RQ"] = rq.item() * 100

        # Thing metrics
        thing_pq, thing_sq, thing_rq = calc_pq_metrics(
            tp[thing_class_idxs].sum(),
            fp[thing_class_idxs].sum(),
            fn[thing_class_idxs].sum(),
            tp_iou[thing_class_idxs].sum(),
        )
        metrics["thing_PQ"] = thing_pq.item() * 100
        metrics["thing_SQ"] = thing_sq.item() * 100
        metrics["thing_RQ"] = thing_rq.item() * 100

        # Stuff metrics
        stuff_pq, stuff_sq, stuff_rq = calc_pq_metrics(
            tp[stuff_class_idxs].sum(),
            fp[stuff_class_idxs].sum(),
            fn[stuff_class_idxs].sum(),
            tp_iou[stuff_class_idxs].sum(),
        )
        metrics["stuff_PQ"] = stuff_pq.item() * 100
        metrics["stuff_SQ"] = stuff_sq.item() * 100
        metrics["stuff_RQ"] = stuff_rq.item() * 100

    # F1 metrics
    if f1_states:
        tp = f1_states["tp_per_class"].float().sum()
        pred = f1_states["pred_per_class"].float().sum()
        gt = f1_states["gt_per_class"].float().sum()
        precision = tp / (pred + eps)
        recall = tp / (gt + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        metrics["F1"] = f1.item()

        w_tp = f1_states["w_tp_per_class"].float().sum()
        w_pred = f1_states["w_pred_per_class"].float().sum()
        w_gt = f1_states["w_gt_per_class"].float().sum()
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
):
    """Run inference on dataset and return results with SymPointV2-compatible save dicts."""
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

                # Accumulate metric states
                if outputs.metric_states is not None:
                    if accumulated_metric_states is None:
                        accumulated_metric_states = {
                            k: v.clone() for k, v in outputs.metric_states.items()
                        }
                    else:
                        for k, v in outputs.metric_states.items():
                            accumulated_metric_states[k] += v

                if outputs.f1_states is not None:
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
    metrics = compute_metrics(accumulated_metric_states, accumulated_f1_states, config)

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
    }


def save_results(results: Dict, output_dir: str, args):
    """Save inference results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics to log file
    log_file = os.path.join(output_dir, f"inference_results_{timestamp}.log")
    with open(log_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("VECFORMER INFERENCE RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.datadir}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"FP16: {args.fp16}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TIMING\n")
        f.write("-" * 60 + "\n")
        timing = results["timing"]
        f.write(f"Total samples: {timing['num_samples']}\n")
        f.write(f"Total inference time: {timing['total_time']:.2f}s\n")
        f.write(f"Average time per sample: {timing['avg_time_per_sample']:.4f}s\n")
        f.write(f"Primitives (min/mean/max/std): {timing['primitives_min']} / {timing['primitives_mean']} / {timing['primitives_max']} / {timing['primitives_std']}\n")
        if "min_time_per_sample" in timing:
            f.write(f"Min time per sample: {timing['min_time_per_sample']:.4f}s\n")
            f.write(f"Max time per sample: {timing['max_time_per_sample']:.4f}s\n")
            f.write(f"Std time per sample: {timing['std_time_per_sample']:.4f}s\n")
            f.write(f"VRAM allocated (min/mean/max/std): {timing['vram_allocated_min_mb']:.1f} / {timing['vram_allocated_mean_mb']:.1f} / {timing['vram_allocated_max_mb']:.1f} / {timing['vram_allocated_std_mb']:.1f} MB\n")
            f.write(f"VRAM reserved  (min/mean/max/std): {timing['vram_reserved_min_mb']:.1f} / {timing['vram_reserved_mean_mb']:.1f} / {timing['vram_reserved_max_mb']:.1f} / {timing['vram_reserved_std_mb']:.1f} MB\n")
            f.write(f"VRAM physical  (min/mean/max/std): {timing['vram_physical_min_mb']:.1f} / {timing['vram_physical_mean_mb']:.1f} / {timing['vram_physical_max_mb']:.1f} / {timing['vram_physical_std_mb']:.1f} MB\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("METRICS\n")
        f.write("-" * 60 + "\n")

        metrics = results["metrics"]
        if "PQ" in metrics:
            f.write("\nPanoptic Segmentation:\n")
            f.write(f"  PQ: {metrics['PQ']:.3f}  |  SQ: {metrics['SQ']:.3f}  |  RQ: {metrics['RQ']:.3f}\n\n")

            f.write("Thing classes:\n")
            f.write(f"  PQ: {metrics['thing_PQ']:.3f}  |  SQ: {metrics['thing_SQ']:.3f}  |  RQ: {metrics['thing_RQ']:.3f}\n\n")

            f.write("Stuff classes:\n")
            f.write(f"  PQ: {metrics['stuff_PQ']:.3f}  |  SQ: {metrics['stuff_SQ']:.3f}  |  RQ: {metrics['stuff_RQ']:.3f}\n\n")

        if "F1" in metrics:
            f.write("Semantic Segmentation:\n")
            f.write(f"  F1:  {metrics['F1']:.4f}\n")
            f.write(f"  wF1: {metrics['wF1']:.4f}\n\n")

        if results["skipped_files"]:
            f.write("-" * 60 + "\n")
            f.write("SKIPPED FILES (OOM)\n")
            f.write("-" * 60 + "\n")
            for skipped in results["skipped_files"]:
                f.write(f"  {skipped}\n")

    logger.info(f"Saved results log to {log_file}")

    # Save metrics as JSON
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "metrics": results["metrics"],
                "timing": results["timing"],
                "skipped_files": results["skipped_files"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics JSON to {metrics_file}")

    # Save model_output.npy (SymPointV2-compatible format)
    if results["save_dicts"]:
        npy_file = os.path.join(output_dir, "model_output.npy")
        np.save(npy_file, results["save_dicts"])
        logger.info(f"Saved {len(results['save_dicts'])} predictions to {npy_file}")


def main():
    args = get_args()

    logger.info("=" * 60)
    logger.info("VecFormer Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.datadir}")
    logger.info(f"Output Dir: {args.out}")
    logger.info(f"Device: {args.device}")
    logger.info(f"FP16: {args.fp16}")

    # Load model
    logger.info("\nLoading model...")
    model, config = load_model(args.checkpoint, args.device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset, collate_fn = load_dataset(args.datadir)
    logger.info(f"Dataset size: {len(dataset)} samples")

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
    )

    # Print metrics
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    metrics = results["metrics"]
    if "PQ" in metrics:
        logger.info(f"PQ: {metrics['PQ']:.3f}  |  SQ: {metrics['SQ']:.3f}  |  RQ: {metrics['RQ']:.3f}")
        logger.info(f"Thing PQ: {metrics['thing_PQ']:.3f}  |  Stuff PQ: {metrics['stuff_PQ']:.3f}")

    if "F1" in metrics:
        logger.info(f"F1: {metrics['F1']:.4f}  |  wF1: {metrics['wF1']:.4f}")

    logger.info(f"\nInference time: {results['timing']['total_time']:.2f}s ({results['timing']['avg_time_per_sample']:.4f}s/sample)")

    if results["skipped_files"]:
        logger.warning(f"Skipped {len(results['skipped_files'])} files due to OOM")

    # Save results
    save_results(results, args.out, args)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
