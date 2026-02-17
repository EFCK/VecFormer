#!/usr/bin/bash
# Fine-tuning script for Poilabs data (made by EFCK)
# Uses vecformer_poilabs.yaml which sets ignore_label to exclude classes
# not present in Poilabs data from evaluation metrics during training.
#
# Usage:
#   ./scripts/train_poilabs.sh
#
# Override data dir:
#   OUTPUT_DIR=outputs/poilabs TIMESTAMP=run1 ./scripts/train_poilabs.sh

export PYTHONPATH=$(pwd):$PYTHONPATH

export TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
export OUTPUT_DIR=${OUTPUT_DIR:-"outputs/poilabs"}

export NODE_RANK=${NODE_RANK:-0}
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-11451}

export OMP_NUM_THREADS=$NPROC_PER_NODE
torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    launch.py \
    --launch_mode train \
    --config_path configs/vecformer.yaml \
    --model_args_path configs/model/vecformer_poilabs.yaml \
    --data_args_path configs/data/floorplancad.yaml \
    --run_name vecformer_poilabs \
    --save_total_limit 5 \
    --output_dir ${OUTPUT_DIR}/${TIMESTAMP}
