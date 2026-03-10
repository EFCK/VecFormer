# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VecFormer is a deep learning model for **panoptic symbol spotting in CAD floor plan drawings** (NeurIPS 2025). The key innovation is using a **line-based representation** (x1, y1, x2, y2 primitives) instead of point-based representations for semantic and instance segmentation of architectural symbols in the FloorPlanCAD dataset (35 symbol categories: 30 thing + 5 stuff classes).

## Environment Setup

```bash
conda create -n vecformer python=3.9 -y && conda activate vecformer
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
# flash-attention must be built from source (see README.md)
pip install -r requirements.txt
```

Always set `PYTHONPATH` before running scripts:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Convenience scripts: `source env.sh` (Linux) or `source env_wsl.sh` (WSL).

## Common Commands

**Training:**
```bash
bash scripts/train.sh
# Outputs saved to: outputs/{TIMESTAMP}/
```

**Evaluation:**
```bash
bash scripts/test.sh
# Uses checkpoint configs/cp_98, results saved to results/98_epoch/
```

**Resume training** (preserves optimizer state):
```bash
bash scripts/resume.sh
```

**Continue training** (loads weights, resets optimizer):
```bash
bash scripts/continue.sh
```

**Inference** (no training loop, direct prediction):
```bash
bash scripts/infer.sh
# Or directly:
python inference.py --checkpoint configs/cp_98 --datadir <path> --out <path> [--profile] [--ignore_mode poilabs]
```

**Data preprocessing:**
```bash
# Line-based (recommended):
python data/floorplancad/preprocess.py \
    --input_dir=$(pwd)/datasets/FloorPlanCAD \
    --output_dir=$(pwd)/datasets/FloorPlanCAD-sampled-as-line-jsons \
    --dynamic_sampling --connect_lines --use_progress_bar

# Point-based (omit --connect_lines):
python data/floorplancad/preprocess.py \
    --input_dir=$(pwd)/datasets/FloorPlanCAD \
    --output_dir=$(pwd)/datasets/FloorPlanCAD-sampled-as-point-jsons \
    --dynamic_sampling --use_progress_bar
```

**Visualization:**
```bash
python svg_visualize_v3.py [options]
```

## Architecture

### Configuration System

Arguments are loaded from a YAML config file and overridden by CLI args. The main training config is `configs/vecformer.yaml`. Model and data configs are passed separately via `--model_args_path` and `--data_args_path`. CLI args that differ from defaults override the YAML values (see `utils/args.py`).

Key config files:
- `configs/vecformer.yaml` — training hyperparameters (epochs, batch size, LR, scheduler, logging)
- `configs/model/vecformer.yaml` — model architecture overrides (uses `VecFormerConfig` defaults)
- `configs/data/floorplancad.yaml` — dataset paths and augmentation settings
- `configs/cp_98/`, `configs/cp_106/` — pretrained checkpoints

### Model Stack

The model is a HuggingFace `PreTrainedModel` built around three main components:

1. **VecBackbone** (`model/vecformer/vec_backbone/`) — Encoder-decoder with absolute position embedding and group fusion, wrapping PointTransformerV3
2. **PointTransformerV3** (`model/vecformer/point_transformer_v3/`) — Processes vector primitives with spatial attention and hierarchical encoding using sparse convolutions (spconv)
3. **CADDecoder** (`model/vecformer/cad_decoder/`) — Transformer-based decoder that produces instance and semantic segmentation outputs

Losses and metrics:
- **Criterion** (`model/vecformer/criterion/`) — instance loss (BCE + Dice + focal) + semantic loss (cross-entropy)
- **Evaluator** (`model/vecformer/evaluator/`) — computes PQ (Panoptic Quality = √(SQ × RQ)), IoU, and per-class metrics; primary eval metric is `PQ`

### Data Pipeline

`data/floorplancad/preprocess.py` converts raw FloorPlanCAD SVGs into JSON files. Each JSON represents a floor plan as a list of line primitives with features: coordinates (x1, y1, x2, y2), length, direction, color, width, and semantic label.

`data/floorplancad/floorplancad.py` is the HuggingFace `Dataset`-compatible loader. `data/floorplancad/dataclass_define.py` defines `SVGData` and `VecData` structures. Batches use variable-length sequences tracked via `cu_seqlens` (cumulative sequence lengths) for efficient Flash Attention processing.

#### Preprocessing (SVG → JSON) — `data/floorplancad/preprocess.py`

1. **parse_color_to_rgb()** — handles rgb(), hex, named, and numeric color fallback
2. **parse_primitive()** — samples path primitives; skips zero-length (`prim_length < 1e-10`); outputs `[x, y]` (point) or `[x1, y1, x2, y2]` (line)
3. **parse_svg()** — produces `SVGData`: `{viewBox, coords, colors, widths, primitive_ids, layer_ids, semantic_ids, instance_ids, primitive_lengths}`
4. **validate_labels()** — logs invalid (semantic, instance) pairs to `logs/<folder>/invalid_labels.log`
5. Output layout (SymPointV2-style): `<base>/line_json/<folder>/`, `point_json/<folder>/`, `svg/`, `logs/`

#### Dataset Loading (JSON → VecData batch) — `data/floorplancad/floorplancad.py`

Transform pipeline per sample:
1. **to_tensor()** — SVGData → SVGDataTensor (torch)
2. **norm_coords()** — coordinates normalized to `[-0.5, 0.5]`
3. **augment_line_args()** — training: h/v flip (p=0.5), scale [0.8–1.2], translate ±0.1; eval: disabled
4. **to_vec_data()** → `VecData`:
   - `coords`: (N, 4) line or (N, 2) point
   - `feats`: (N, 10) for lines = `[length, |dx|, |dy|, cx, cy, pcx, pcy, r, g, b]`
     - `pcx/pcy` = primitive center via `scatter(..., reduce="mean")`
     - RGB normalized: `color/255 - 0.5`

**Batch collation** (`collate_fn`): concatenates all samples along N; tracks variable lengths via:
- `cu_seqlens`: (B+1,) cumulative primitive counts (for Flash Attention)
- `cu_numprims`: (B+1,) cumulative unique primitive counts

#### Model Forward Pass — `model/vecformer/modeling_vecformer.py`

1. **prepare_targets()** — filters invalid (sem_id, inst_id) pairs; builds instance masks (M, N) and panoptic masks
2. **_get_data_dict()** — adds z=0 (point) or layer_id (line) as extra feature; sets `offset = cu_seqlens[1:]`
3. **PointTransformerV3 backbone** — hierarchical sparse convolution → feature embeddings
4. **Layer fusion (lfe)** — optional: fuses layer_id information into backbone features via `FusionLayerFeatsModule`
5. **_init_queries()** — training: random subset (~`query_thr` fraction, capped by `max_num_queries`); eval: all features
6. **CADDecoder** (3 blocks by default):
   - Self-attention on queries → cross-attention queries→features → FFN
   - Per-block heads: semantic logits (Q, C+1), instance logits (Q, C+1), objectness scores (Q, 1), instance masks (Q, N) via `queries @ mask_feats^T`

#### Postprocessing (Logits → Predictions) — `model/vecformer/modeling_vecformer.py`

**predict_semantic()**: softmax → argmax on primitive-level logits

**predict_instance()** pipeline:
1. Softmax + argmax on instance labels
2. Score = label_prob × objectness_score
3. Top-k filtering by score
4. Sigmoid threshold on mask logits (`> mask_logit_thr`)
5. Object normalization (optional): adjust score by mask coverage ratio
6. Score threshold filter (`<= pred_score_thr` removed)
7. Sparsity filter (`< n_primitives_thr` removed)

**predict_panoptic()**:
1. Voting: reassign instance semantic labels by majority vote from primitive semantics
2. Remasking: remove primitives from instance where semantic ≠ instance label
3. Stuff: create masks from semantic labels for stuff classes
4. Concat instance + stuff masks = final panoptic output

#### Evaluation — `model/vecformer/evaluator/evaluator.py`

- **IoU**: vectorized (N_targets × N_preds) matrix, weighted by `log(1 + prim_len)`
- **PQ = SQ × RQ** = `(tp_iou_sum / tp) × (tp / (tp + 0.5×fp + 0.5×fn))`
- Computed separately for things/stuff/total; ignored classes excluded via `torch.isin()`

#### Inference Output — `inference.py`

`save_results()` writes:
- `inference_results_*.log` — per-class PQ, mIoU, GT status
- `metrics_*.json` — compact metrics dict
- `model_output.npy` — SymPointV2-compatible: `{filepath, sem, ins, targets, lengths}`

### Entry Points

- `launch.py` — Main entry point for training/evaluation via `torchrun`. Supports `--launch_mode train|test`.
- `inference.py` — Standalone inference without the training loop; supports `--profile` for per-sample memory stats and `--ignore_mode poilabs` for custom label filtering.

### Key Utilities

- `utils/patches.py` — Applied at startup to fix library compatibility issues (must be imported first)
- `utils/parallel_mapper.py` — Parallel data processing for preprocessing
- `utils/svg_util.py` — SVG parsing

## Important Notes

- The model uses **Flash Attention** and **sparse convolutions** (spconv-cu118). Both require CUDA 11.8+.
- Distributed training uses `torchrun`; environment variables `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` control the setup.
- `utils/patches.py` must be imported before other modules in any new entry point.
- The `ignore_mode poilabs` flag in inference enables evaluation on the Poilabs dataset variant where some classes are ignored.
- Logs are written to `logs/<input_folder>/` during preprocessing.

## Key Customizations

This section records non-obvious departures from upstream HuggingFace and SymPointV2 conventions, so future instances don't accidentally revert or misunderstand them.

### HuggingFace Patches (`utils/patches/`)

Applied at startup via `utils/patches.py` → `apply_patches()`. This module **must be imported before any other modules** in every entry point.

- **logging_patch.py**: Replaces HF's log formatter with one that includes timestamps and `filename:line` — makes training logs scannable.
- **printer_callback_patch.py**: Adapted from ms-swift — adds elapsed/remaining time logging, saves eval metrics to `logging.jsonl`, and creates symlinks `outputs/latest/checkpoint-best` and `outputs/latest/checkpoint-latest` after each eval checkpoint.
- **training_arguments_patch.py**: Adds `config_path`, `model_args_path`, `data_args_path`, and `launch_mode` fields to HF `TrainingArguments` so the YAML config system integrates cleanly with HF's `Trainer`.

### SymPointV2-Compatible Interfaces

- **Preprocessing output layout**: `preprocess.py` writes to `<base>/line_json/<folder>/` (not into the input directory) — matches the SymPointV2 dataset directory structure so tooling is interchangeable.
- **Inference output format**: `build_save_dict()` in `inference.py` produces `model_output.npy` in SymPointV2-compatible format (`sem`, `ins`, `targets`, `lengths`) for downstream evaluation scripts.

### `ignore_label`: `int` → `list[int]`

`VecFormerConfig`, `EvaluatorConfig`, `MetricsComputer`, and `Evaluator` all accept `ignore_label` as `list[int]` (not a single int). This supports multi-class ignore (e.g., `[35]` for background-only, or a longer list for the Poilabs variant). All `== ignore_label` comparisons use `torch.isin()`. The Poilabs variant is configured via `configs/model/vecformer_poilabs.yaml` with a custom ignore list.

### Trainer `ignore_label` Fix (`vecformer_trainer.py`)

`MetricsComputer` is initialized from `kwargs["model"].config` (the actual loaded model config), **not** from `VecFormerConfig()` defaults. Without this fix, `ignore_label` is always `[35]` regardless of what the YAML specifies — silently wrong for Poilabs runs.

### Evaluator Vectorization (`evaluator.py`)

IoU computation and metric aggregation are fully vectorized using batched tensor ops, `scatter_add`, and batched `.item()` calls. Do **not** reintroduce Python loops over target/prediction pairs — they were deliberately removed for GPU efficiency.
