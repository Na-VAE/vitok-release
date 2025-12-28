"""Example config: Train DiT on ImageNet using HuggingFace streaming.

This config streams ImageNet from HuggingFace Hub without downloading the full dataset.

Usage:
    python scripts/train_dit.py --config examples/configs/imagenet_streaming.py
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class config:
    """Training config for ImageNet streaming."""

    # Model
    ae_variant: str = "Ld2-Ld22/1x16x64"  # Large encoder/decoder
    ae_checkpoint: str = "path/to/ae.safetensors"  # Replace with your checkpoint
    dit_variant: str = "L/256"  # Large DiT, 256 tokens (16x16 grid)
    dit_checkpoint: Optional[str] = None  # Set to resume training
    num_classes: int = 1000

    # Data - HuggingFace ImageNet streaming
    hf_repo: str = "ILSVRC/imagenet-1k"  # HF ImageNet repo
    data_paths: List[str] = field(default_factory=list)  # Not used for HF streaming
    batch_size: int = 256  # Per-GPU batch size
    num_workers: int = 8
    max_tokens: int = 256  # 16x16 = 256 tokens
    image_size: int = 256

    # Training
    steps: int = 400000
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    cfg_dropout: float = 0.1

    # EMA
    ema_decay: float = 0.9999
    ema_start_step: int = 5000

    # Logging & Checkpointing
    log_freq: int = 100
    sample_freq: int = 5000
    save_freq: int = 10000
    output_dir: str = "checkpoints/dit_imagenet"
    wandb_project: str = "vitok-dit"
    wandb_name: str = "imagenet-L256"

    # System
    seed: int = 42
    compile: bool = True  # Use torch.compile for speed
    fsdp: bool = True  # Use FSDP for multi-GPU
    bf16: bool = True
