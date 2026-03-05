"""Deterministic scatter operations using PyTorch native ops.

Replaces torch_scatter.scatter with torch.scatter_reduce_ which respects
torch.use_deterministic_algorithms(True). The torch_scatter library uses
custom CUDA kernels with atomic operations that bypass PyTorch's
determinism checks, causing run-to-run variance.
"""

import torch


# Map torch_scatter reduce names to PyTorch scatter_reduce_ names
_REDUCE_MAP = {
    "sum": "sum",
    "mean": "mean",
    "max": "amax",
    "min": "amin",
}


def scatter(src, index, dim=0, out=None, dim_size=None, fill_value=0, reduce="sum"):
    """Drop-in replacement for torch_scatter.scatter using PyTorch native ops.

    Args:
        src: Source tensor.
        index: Index tensor (same length as src along dim).
        dim: Dimension along which to scatter.
        out: Unused (kept for API compatibility).
        dim_size: Size of output along dim. If None, uses index.max()+1.
        fill_value: Fill value for positions not written to.
        reduce: Reduction operation ("sum", "mean", "max", "min").
    """
    if dim != 0:
        raise NotImplementedError("Only dim=0 is supported")

    pt_reduce = _REDUCE_MAP.get(reduce)
    if pt_reduce is None:
        raise ValueError(f"Unsupported reduce: {reduce}")

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    # Expand index to match src shape
    if src.dim() > 1 and index.dim() == 1:
        index = index.unsqueeze(-1).expand_as(src)

    # Build output tensor with fill_value for unwritten positions
    shape = list(src.shape)
    shape[dim] = dim_size
    result = src.new_full(shape, fill_value)

    # include_self=False: only scattered values participate in the reduction.
    # Unwritten positions keep their initial fill_value.
    result.scatter_reduce_(dim, index, src, reduce=pt_reduce, include_self=False)
    return result
