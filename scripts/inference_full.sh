#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# Default Configuration
# ==========================================
DEFAULT_INPUT_DIR="./datasets/FinetuneV2_bf"
DEFAULT_OUTPUT_NAME="FinetuneV2"
DEFAULT_CHECKPOINT="configs/cp_192"
DEFAULT_IGNORE_MODE="poilabs"
# ==========================================

# Initialize variables
INPUT_DIR="${DEFAULT_INPUT_DIR}"
OUTPUT_NAME="${DEFAULT_OUTPUT_NAME}"
CHECKPOINT="${DEFAULT_CHECKPOINT}"
IGNORE_MODE="${DEFAULT_IGNORE_MODE}"

# Helper function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i  Input dataset directory (default: ${DEFAULT_INPUT_DIR})"
    echo "  -n  Output dataset name (default: ${DEFAULT_OUTPUT_NAME})"
    echo "  -c  Path to checkpoint file or directory (default: ${DEFAULT_CHECKPOINT})"
    echo "  -m  Ignore mode (default: ${DEFAULT_IGNORE_MODE})"
    echo "  -h  Show this help message"
    echo ""
    echo "If no options are provided, the script uses the default variables defined inside."
    exit 1
}

# Parse command line options
while getopts i:n:c:m:h flag; do
    case "${flag}" in
        i) INPUT_DIR="${OPTARG}" ;;
        n) OUTPUT_NAME="${OPTARG}" ;;
        c) CHECKPOINT="${OPTARG}" ;;
        m) IGNORE_MODE="${OPTARG}" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "=========================================="
echo "Inference Pipeline Configuration:"
echo "------------------------------------------"
echo "Input Directory : ${INPUT_DIR}"
echo "Output Name     : ${OUTPUT_NAME}"
echo "Checkpoint      : ${CHECKPOINT}"
echo "Ignore Mode     : ${IGNORE_MODE}"
echo "=========================================="
echo ""

# 1. Reshape
echo "[1/4] Reshaping SVGs from ${INPUT_DIR}..."
python svg_reshape_v2.py --input "${INPUT_DIR}" --output "./datasets/${OUTPUT_NAME}/" --ignore_mode "${IGNORE_MODE}"
echo "Done."
echo ""

# 2. Preprocess
echo "[2/4] Preprocessing Data..."
python data/floorplancad/preprocess.py \
    --input_dir="./datasets/${OUTPUT_NAME}" \
    --dynamic_sampling \
    --connect_lines \
    --use_progress_bar
echo "Done."
echo ""

# 3. Inference
echo "[3/4] Running Inference..."
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Directories for inference step
DATADIR="datasets/line_json/${OUTPUT_NAME}/"
OUT="results/${OUTPUT_NAME}/"

python inference.py \
    --checkpoint "${CHECKPOINT}" \
    --datadir "${DATADIR}" \
    --out "${OUT}" \
    --profile \
    --ignore_mode "${IGNORE_MODE}"
echo "Done."
echo ""

# 4. Visualize
echo "[4/4] Visualizing Results..."
python svg_visualize_v3.py --res_file "./results/${OUTPUT_NAME}/" --semantic --perfile --ignore_mode "${IGNORE_MODE}"
echo "Done."
echo ""

echo "=========================================="
echo "Inference pipeline completed successfully."
echo "Results available in: ./results/${OUTPUT_NAME}/"
echo "=========================================="
