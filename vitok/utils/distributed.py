"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


def setup_distributed(seed: int = 42):
    """Setup distributed training if available.

    Returns:
        (rank, world_size, device) tuple
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Set seed for reproducibility
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + rank)

        return rank, world_size, device

    # Single GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return 0, 1, device


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = ["setup_distributed", "cleanup_distributed"]
