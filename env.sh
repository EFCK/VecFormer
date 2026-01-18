#!/usr/bin/env bash
# VecFormer Environment Activation Script for Fedora
# Usage: source env.sh
#
# This script activates the vecformer conda environment with proper CUDA settings.
# If the environment doesn't exist, it will create it and install all dependencies.

# Detect if being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced, not executed."
    echo "Usage: source env.sh"
    exit 1
fi

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if vecformer environment exists
if ! conda env list | grep -q "^vecformer "; then
    echo "=== VecFormer environment not found. Creating it now... ==="
    echo "This will take a while (building flash-attention from source)."
    echo ""
    
    # Create new environment with Python 3.9
    conda create -n vecformer python=3.9 -y
    conda activate vecformer
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install PyTorch 2.5.1 + cu118
    echo "=== Installing PyTorch 2.5.1 with CUDA 11.8 ==="
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    
    # Install CUDA toolkit via conda
    echo "=== Installing CUDA toolkit via conda ==="
    conda install -y -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-toolkit
    
    # Install GCC compiler
    echo "=== Installing compatible GCC compiler ==="
    conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
    
    # Set environment variables for building
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    export CUDAHOSTCXX="$CXX"
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
    
    # Install build dependencies
    echo "=== Installing build dependencies ==="
    pip install packaging ninja psutil
    
    # Install torch-scatter
    echo "=== Installing torch-scatter ==="
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
    
    # Install flash-attention (use prebuilt wheel for torch 2.3 which works with 2.5)
    echo "=== Installing flash-attention ==="
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
    
    # Install project requirements
    echo "=== Installing project requirements ==="
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    echo ""
    echo "=== Environment setup complete ==="
else
    # Just activate the existing environment
    conda activate vecformer
fi

# Set CUDA environment variables
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Set compilers for any CUDA extensions that need to be built
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"

# Set target GPU architecture (adjust for your GPU if needed)
# Common values: 7.5 (RTX 20xx), 8.0 (A100), 8.6 (RTX 30xx), 8.9 (RTX 40xx)
export TORCH_CUDA_ARCH_LIST="8.6"

# Print summary
python - <<'PYCODE'
import torch
import os

print("\n" + "="*50)
print("VecFormer environment activated")
print("="*50)
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print("="*50 + "\n")
PYCODE
