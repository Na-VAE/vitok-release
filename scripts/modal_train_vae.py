#!/usr/bin/env python
"""Modal script for debugging VAE training on ImageNet-22k.

Usage:
    # First time: sync code to volume (required before first run)
    modal run scripts/modal_train_vae.py --sync-only

    # Run training (after syncing code)
    modal run scripts/modal_train_vae.py --steps 100

    # Sync code AND run training
    modal run scripts/modal_train_vae.py --sync --steps 100

    # Custom settings
    modal run scripts/modal_train_vae.py --steps 1000 --log-freq 50 --eval-freq 200

    # With wandb logging
    modal run scripts/modal_train_vae.py --wandb-project vitok-debug

The base image (PyTorch, dependencies) is cached by Modal after first build (~5 min).
Code is stored in a Modal Volume and synced separately for fast iteration.
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("vitok-vae-train")

# Get local vitok directory
LOCAL_VITOK_DIR = Path(__file__).parent.parent
LOCAL_DINO_DIR = LOCAL_VITOK_DIR.parent / "dino_perceptual"

# Base image with just dependencies (cacheable)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "numpy",
        "scipy",
        "tqdm",
        "wandb",
        "safetensors",
        "webdataset",
        "torchmetrics",
        "Pillow",
        "huggingface_hub",
        "transformers",
        "pytorch_fid",
    )
)

# Volume for code (allows fast updates without rebuilding image)
code_volume = modal.Volume.from_name("vitok-code", create_if_missing=True)

# Volume for checkpoints
checkpoint_volume = modal.Volume.from_name("vitok-checkpoints", create_if_missing=True)


@app.function(image=base_image, volumes={"/code": code_volume})
def sync_code():
    """Sync local code to Modal volume. Run after code changes."""
    import shutil
    import os

    # This function receives code via the volumes mount
    # The actual sync happens via local_entrypoint below
    print("Code volume mounted at /code")
    print("Contents:")
    for item in os.listdir("/code"):
        print(f"  {item}")
    return "Sync complete"


def _sync_code_to_volume():
    """Helper to sync local code to Modal volume."""
    import subprocess
    import shutil
    import tempfile

    print("=" * 60)
    print("Syncing code to Modal volume...")
    print("=" * 60)

    # Sync vitok-release
    print(f"\nSyncing vitok-release from {LOCAL_VITOK_DIR}...")

    # Use modal volume put to sync code
    # First, create a clean copy without __pycache__ etc
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy vitok-release
        vitok_tmp = Path(tmpdir) / "vitok-release"
        shutil.copytree(
            LOCAL_VITOK_DIR,
            vitok_tmp,
            ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '.git', '.venv*', 'wandb',
                '*.egg-info', '.pytest_cache', '.ruff_cache'
            )
        )

        # Copy dino_perceptual
        dino_tmp = Path(tmpdir) / "dino_perceptual"
        if LOCAL_DINO_DIR.exists():
            shutil.copytree(
                LOCAL_DINO_DIR,
                dino_tmp,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.venv*')
            )
            print(f"Syncing dino_perceptual from {LOCAL_DINO_DIR}...")

        # Upload to volume with --force to overwrite existing files
        print("\nUploading to Modal volume 'vitok-code'...")
        subprocess.run(["modal", "volume", "put", "vitok-code", str(vitok_tmp), "/", "--force"], check=True)
        if dino_tmp.exists():
            subprocess.run(["modal", "volume", "put", "vitok-code", str(dino_tmp), "/", "--force"], check=True)

    print("Code sync complete!")


@app.function(
    image=base_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    volumes={
        "/checkpoints": checkpoint_volume,
        "/code": code_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_vae_debug(
    steps: int = 100000,
    log_freq: int = 100,
    eval_freq: int = 5000,
    batch_size: int = 256,
    wandb_project: str = None,
    wandb_name: str = None,
    debug: bool = False,
):
    """Run VAE training on ImageNet-22k.

    Args:
        steps: Training steps (default: 100k for full run)
        log_freq: Logging frequency (default: 100)
        eval_freq: Evaluation frequency (default: 5000)
        batch_size: Batch size (default: 256, requires A100-80GB)
        wandb_project: WandB project name
        wandb_name: WandB run name
        debug: Use debug settings (small batch, no compile, fewer shards)
    """
    import subprocess
    import os

    # Set Python path to use code from volume
    os.environ["PYTHONPATH"] = "/code/vitok-release:/code/dino_perceptual"

    # Change to vitok directory
    os.chdir("/code/vitok-release")

    # Data source - use all shards for production, fewer for debug
    if debug:
        data_source = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar"
    else:
        data_source = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar"

    # Build command
    cmd = [
        "python", "scripts/train_vae.py",
        # Data - use brace expansion to avoid HfFileSystem.glob() stalls
        "--data", data_source,
        # Training params
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--max_size", "512",
        "--max_tokens", "256",
        # Logging
        "--log_freq", str(log_freq),
        "--eval_freq", str(eval_freq),
        "--output_dir", "/checkpoints/vae",
        "--save_freq", "5000",
        "--marked_freq", "0",  # Disabled - use save_freq checkpoints only
    ]

    # Debug mode: disable compile for faster startup
    if debug:
        cmd.append("--no_compile")

    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_name:
        cmd.extend(["--wandb_name", wandb_name])

    print(f"Running command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run training with unbuffered output
    import sys
    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return result.returncode


@app.local_entrypoint()
def main(
    steps: int = 100000,
    log_freq: int = 100,
    eval_freq: int = 5000,
    batch_size: int = 256,
    wandb_project: str = None,
    wandb_name: str = None,
    sync: bool = False,
    sync_only: bool = False,
    debug: bool = False,
):
    """Local entrypoint for Modal.

    Args:
        steps: Number of training steps (default: 100k)
        log_freq: Logging frequency (default: 100)
        eval_freq: Evaluation frequency (default: 5000)
        batch_size: Batch size per GPU (default: 256)
        wandb_project: Optional wandb project name
        wandb_name: Optional wandb run name
        sync: Sync code before running (default: False)
        sync_only: Only sync code, don't run training (default: False)
        debug: Debug mode with smaller settings (default: False)
    """
    # Sync code if requested
    if sync or sync_only:
        _sync_code_to_volume()
        if sync_only:
            print("\nCode synced. Run training with:")
            print("  modal run scripts/modal_train_vae.py")
            return

    # Debug mode overrides
    if debug:
        batch_size = min(batch_size, 8)
        eval_freq = min(eval_freq, 50)
        log_freq = min(log_freq, 10)

    print("=" * 60)
    print("ViTok VAE Training on Modal")
    print("=" * 60)
    print(f"  Model: Ld2-Ld22/1x16x64")
    print(f"  GPU: A100-80GB")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Log freq: {log_freq}")
    print(f"  Eval freq: {eval_freq}")
    print(f"  Train data: ImageNet-22k ({'50 shards' if debug else '1024 shards'})")
    print(f"  Compile: {'No' if debug else 'Yes'}")
    if wandb_project:
        print(f"  WandB: {wandb_project}/{wandb_name or 'auto'}")
    print("=" * 60)
    print()

    # Estimate cost (with compile: ~0.3 sec/step at bs=256)
    sec_per_step = 0.3 if not debug else 0.8
    est_time_min = steps * sec_per_step / 60
    est_cost = est_time_min / 60 * 4.50  # $4.50/hr for A100
    print(f"Estimated time: ~{est_time_min:.0f} min")
    print(f"Estimated cost: ~${est_cost:.2f}")
    print()

    result = train_vae_debug.remote(
        steps=steps,
        log_freq=log_freq,
        eval_freq=eval_freq,
        batch_size=batch_size,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        debug=debug,
    )

    print(f"Training completed with return code: {result}")
