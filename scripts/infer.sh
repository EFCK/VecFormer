#!/usr/bin/env bash
# VecFormer Inference Script
# Usage: ./scripts/infer.sh
#
# All configurable parameters are defined below.
# Modify these values before running inference.

export PYTHONPATH=$(pwd):$PYTHONPATH

# ==================== Configuration ==================== #
# Dataset directory (contains train/val/test subdirectories)
datadir=datasets/line_json/ManualVsAuto/test/manual

# Output directory for resu""""""""""   lts
out=results/ManualVsAuto/test/manual

# Model checkpoint directory (contains model.safetensors)
checkpoint=configs/cp_98



# ======================================================= #

python inference.py \
    --checkpoint $checkpoint \
    --datadir $datadir \
    --out $out \
    --profile \
    --ignore_mode poilabs
