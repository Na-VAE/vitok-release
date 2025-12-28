"""Example config: Train DiT on custom tar shards.

This config trains on local WebDataset tar files.

Tar file format:
    Each sample should contain:
    - {key}.jpg (or .png, .webp): The image
    - {key}.cls (optional): Class label as integer

Usage:
    python scripts/train_dit.py --config examples/configs/generic_tar.py \
        --data_paths /path/to/shards/

Directory structure example:
    /data/my_dataset/
        shard-00000.tar
        shard-00001.tar
        ...
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class config:
    """Training config for generic tar shards."""

    # Model - use smaller variant for custom datasets
    ae_variant: str = "Ld2-Ld22/1x16x64"
    ae_checkpoint: str = "path/to/ae.safetensors"  # Replace with your checkpoint
    dit_variant: str = "B/256"  # Base DiT for smaller datasets
    dit_checkpoint: Optional[str] = None
    num_classes: int = 100  # Adjust based on your dataset

    # Data - local tar shards
    data_paths: List[str] = field(default_factory=lambda: ["/path/to/shards"])
    hf_repo: Optional[str] = None  # Not using HF
    batch_size: int = 64  # Smaller batch for limited GPU memory
    num_workers: int = 4
    max_tokens: int = 256
    image_size: int = 256

    # Training - shorter for smaller datasets
    steps: int = 100000
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    cfg_dropout: float = 0.1

    # EMA
    ema_decay: float = 0.9999
    ema_start_step: int = 2000

    # Logging & Checkpointing
    log_freq: int = 50
    sample_freq: int = 2000
    save_freq: int = 5000
    output_dir: str = "checkpoints/dit_custom"
    wandb_project: Optional[str] = None  # Set to enable W&B logging
    wandb_name: Optional[str] = None

    # System
    seed: int = 42
    compile: bool = False  # Disable for debugging
    fsdp: bool = False  # Single GPU by default
    bf16: bool = True
