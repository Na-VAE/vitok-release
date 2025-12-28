"""Dtype utilities."""

import torch


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert dtype string to torch.dtype.

    Args:
        dtype: Either a torch.dtype or a string like "bf16", "bfloat16",
               "fp16", "float16", "half", "fp32", "float32", "float".

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If the dtype string is not recognized.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


__all__ = ["resolve_dtype"]
