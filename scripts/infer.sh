#!/usr/bin/env bash
# VecFormer Inference Script
# Usage: ./scripts/infer.sh
#
# All configurable parameters are defined below.
# Modify these values before running inference.

export PYTHONPATH=$(pwd):$PYTHONPATH

# ==================== Configuration ==================== #
# Dataset directory (contains train/val/test subdirectories)
datadir=datasets/FloorPlanCAD-jsons/test

# Model checkpoint directory (contains model.safetensors)
checkpoint=configs/cp_98

# Output directory for results
out=results/floorplancad_test/cp_98/


# ======================================================= #

python inference.py \
    --checkpoint $checkpoint \
    --datadir $datadir \
    --out $out \
    --profile
