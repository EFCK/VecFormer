#!/usr/bin/env bash
# VecFormer Environment Activation Script for Windows WSL
# Usage: source env_wsl.sh
#
# This script activates the vecformer conda environment with proper CUDA settings.
# If the environment doesn't exist, it will create it and install all dependencies.
#
# Differences from Fedora version:
# - Adapted for WSL2 environment (uses Windows GPU drivers)
# - Auto-detects GPU architecture
# - Uses compatible CUDA/PyTorch versions for newer GPUs

# Detect if being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced, not executed."
    echo "Usage: source env_wsl.sh"
    exit 1
fi

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Warning: This script is designed for WSL. Use env.sh for native Linux."
fi

# Check if nvidia-smi works (Windows drivers should provide this)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed on Windows."
    return 1
fi

# Auto-detect GPU compute capability
detect_gpu_arch() {
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    
    if [[ -z "$compute_cap" ]]; then
        echo "8.6"  # Default to RTX 30xx
        return
    fi
    
    # Convert compute capability (e.g., "8.6" stays as "8.6", "12.0" becomes "12.0")
    echo "$compute_cap"
}

GPU_ARCH=$(detect_gpu_arch)
echo "Detected GPU compute capability: $GPU_ARCH"

# Source conda
if [[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda installation."
    echo "Please install miniconda: https://docs.conda.io/en/latest/miniconda.html"
    return 1
fi

# Check if vecformer environment exists
if ! conda env list | grep -q "^vecformer "; then
    echo "=== VecFormer environment not found. Creating it now... ==="
    echo "This will take a while (installing dependencies)."
    echo ""
    
    # Create new environment with Python 3.9
    conda create -n vecformer python=3.9 -y
    
    # Activate environment (need to source conda.sh again for activation to work)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vecformer
    
    if [[ -z "$CONDA_PREFIX" ]] || [[ "$CONDA_DEFAULT_ENV" != "vecformer" ]]; then
        echo "Error: Failed to activate vecformer environment."
        echo "Please run: conda activate vecformer && source env_wsl.sh"
        return 1
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Determine PyTorch/CUDA version based on GPU architecture
    # RTX 50xx (Blackwell, compute 12.0) needs CUDA 12.x
    # RTX 40xx (Ada, compute 8.9) works with CUDA 11.8 or 12.x
    # RTX 30xx (Ampere, compute 8.6) works with CUDA 11.8
    
    MAJOR_ARCH=$(echo "$GPU_ARCH" | cut -d. -f1)
    
    if [[ "$MAJOR_ARCH" -ge 12 ]]; then
        # Blackwell (RTX 50xx) - needs CUDA 12.8+
        echo "=== Detected Blackwell GPU (RTX 50xx). Installing PyTorch with CUDA 12.8 ==="
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        CUDA_VERSION="12.8"
        TORCH_SCATTER_URL="https://data.pyg.org/whl/torch-2.7.0+cu128.html"
    elif [[ "$MAJOR_ARCH" -ge 9 ]] || [[ "$GPU_ARCH" == "8.9" ]]; then
        # Ada Lovelace (RTX 40xx) - use CUDA 12.4 for better compatibility
        echo "=== Detected Ada Lovelace GPU (RTX 40xx). Installing PyTorch with CUDA 12.4 ==="
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        CUDA_VERSION="12.4"
        TORCH_SCATTER_URL="https://data.pyg.org/whl/torch-2.5.0+cu124.html"
    else
        # Ampere/Turing (RTX 30xx/20xx) - use CUDA 11.8
        echo "=== Installing PyTorch 2.5.1 with CUDA 11.8 ==="
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
        CUDA_VERSION="11.8"
        TORCH_SCATTER_URL="https://data.pyg.org/whl/torch-2.5.0+cu118.html"
    fi
    
    # Install CUDA toolkit via conda (matching version)
    echo "=== Installing CUDA toolkit via conda ==="
    if [[ "$CUDA_VERSION" == "12.8" ]]; then
        conda install -y -c nvidia/label/cuda-12.8.0 cuda-nvcc cuda-toolkit 2>/dev/null || \
        conda install -y -c nvidia cuda-nvcc cuda-toolkit
    elif [[ "$CUDA_VERSION" == "12.4" ]]; then
        conda install -y -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-toolkit
    else
        conda install -y -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-toolkit
    fi
    
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
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    
    # Install build dependencies
    echo "=== Installing build dependencies ==="
    pip install packaging ninja psutil
    
    # Install torch-scatter
    echo "=== Installing torch-scatter ==="
    pip install torch-scatter -f "$TORCH_SCATTER_URL" || \
    pip install torch-scatter  # Fallback to building from source
    
    # Install flash-attention
    echo "=== Installing flash-attention ==="
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        # Use prebuilt wheel for CUDA 11.8
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl 2>/dev/null || \
        pip install flash-attn --no-build-isolation
    else
        # For CUDA 12.x, install from PyPI or build from source
        pip install flash-attn --no-build-isolation 2>/dev/null || {
            echo "=== Building flash-attention from source (this may take a while) ==="
            pip install flash-attn --no-build-isolation --no-cache-dir
        }
    fi
    
    # Install project requirements
    echo "=== Installing project requirements ==="
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    # Handle spconv version based on CUDA
    if [[ "$CUDA_VERSION" != "11.8" ]]; then
        echo "=== Reinstalling spconv for CUDA $CUDA_VERSION ==="
        pip uninstall -y spconv-cu118 2>/dev/null
        if [[ "$CUDA_VERSION" == "12.8" ]]; then
            pip install spconv-cu128 2>/dev/null || pip install spconv-cu124
        elif [[ "$CUDA_VERSION" == "12.4" ]]; then
            pip install spconv-cu124 2>/dev/null || pip install spconv-cu120
        fi
    fi
    
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

# Set target GPU architecture (auto-detected)
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

# Set PYTHONPATH for the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Print summary
python - <<'PYCODE'
import torch
import os

print("\n" + "="*50)
print("VecFormer environment activated (WSL)")
print("="*50)
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Compute Capability: {props.major}.{props.minor}")
else:
    print("WARNING: CUDA not available!")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")
print("="*50 + "\n")
PYCODE
