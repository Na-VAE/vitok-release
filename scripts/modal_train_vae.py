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


# 8xA100 function for production training
@app.function(
    image=base_image,
    gpu="A100:8",
    timeout=86400,  # 24 hours
    volumes={
        "/checkpoints": checkpoint_volume,
        "/code": code_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_vae(
    steps: int = 100000,
    log_freq: int = 100,
    eval_freq: int = 5000,
    batch_size: int = 64,  # Per GPU, total = 64*8 = 512
    wandb_project: str = None,
    wandb_name: str = None,
):
    """Run VAE training on 8xA100 with FSDP."""
    import subprocess
    import os

    os.environ["PYTHONPATH"] = "/code/vitok-release:/code/dino_perceptual"
    os.chdir("/code/vitok-release")

    data_source = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar"

    # Use torchrun for multi-GPU
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29500",
        "scripts/train_vae.py",
        "--data", data_source,
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--max_size", "512",
        "--max_tokens", "256",
        "--log_freq", str(log_freq),
        "--eval_freq", str(eval_freq),
        "--output_dir", "/checkpoints/vae",
        "--save_freq", "5000",
        "--fsdp",  # Enable FSDP for multi-GPU
    ]

    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_name:
        cmd.extend(["--wandb_name", wandb_name])

    print(f"Running on 8xA100 with FSDP:")
    print(f"  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return result.returncode


# 4xA10G function for debug/testing (cheaper than A100)
@app.function(
    image=base_image,
    gpu="A10G:4",
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
    steps: int = 5000,
    log_freq: int = 50,
    eval_freq: int = 500,
    batch_size: int = 64,  # Per GPU, total = 64*4 = 256
    wandb_project: str = None,
    wandb_name: str = None,
):
    """Run VAE training on 4xA10G with FSDP (for testing/debug)."""
    import subprocess
    import os

    os.environ["PYTHONPATH"] = "/code/vitok-release:/code/dino_perceptual"
    os.chdir("/code/vitok-release")

    data_source = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar"

    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "--master_port=29500",
        "scripts/train_vae.py",
        "--data", data_source,
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--max_size", "512",
        "--max_tokens", "256",
        "--log_freq", str(log_freq),
        "--eval_freq", str(eval_freq),
        "--output_dir", "/checkpoints/vae-debug",
        "--save_freq", "5000",
        "--fsdp",  # Enable FSDP for multi-GPU
    ]

    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_name:
        cmd.extend(["--wandb_name", wandb_name])

    print(f"Running on 4xA10G with FSDP:")
    print(f"  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return result.returncode


@app.local_entrypoint()
def main(
    steps: int = 100000,
    log_freq: int = 100,
    eval_freq: int = 5000,
    batch_size: int = 64,  # Per GPU, total = 64*8 = 512
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
        batch_size: Batch size per GPU (default: 64, total 512 on 8xA100)
        wandb_project: Optional wandb project name
        wandb_name: Optional wandb run name
        sync: Sync code before running (default: False)
        sync_only: Only sync code, don't run training (default: False)
        debug: Debug mode - uses 4xA10G with smaller dataset (default: False)
    """
    # Sync code if requested
    if sync or sync_only:
        _sync_code_to_volume()
        if sync_only:
            print("\nCode synced. Run training with:")
            print("  modal run scripts/modal_train_vae.py")
            return

    n_gpus = 4 if debug else 8
    total_batch = batch_size * n_gpus

    print("=" * 60)
    print("ViTok VAE Training on Modal")
    print("=" * 60)
    print(f"  Model: Ld2-Ld22/1x16x64")
    print(f"  GPU: {'4xA10G (debug)' if debug else '8xA100'}")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size} x {n_gpus} = {total_batch}")
    print(f"  Log freq: {log_freq}")
    print(f"  Eval freq: {eval_freq}")
    print(f"  Train data: ImageNet-22k ({'50 shards' if debug else '1024 shards'})")
    print(f"  FSDP: Yes")
    if wandb_project:
        print(f"  WandB: {wandb_project}/{wandb_name or 'auto'}")
    print("=" * 60)
    print()

    # Estimate cost
    if debug:
        # 4xA10G: ~$4.40/hr (~$1.10/hr each), ~0.15 sec/step at bs=256
        sec_per_step = 0.15
        hourly_rate = 4.40
    else:
        # 8xA100-40GB: ~$24/hr (~$3/hr each), ~0.05 sec/step at bs=512
        sec_per_step = 0.05
        hourly_rate = 24.0

    est_time_min = steps * sec_per_step / 60
    est_cost = est_time_min / 60 * hourly_rate
    print(f"Estimated time: ~{est_time_min:.0f} min")
    print(f"Estimated cost: ~${est_cost:.2f}")
    print()

    if debug:
        result = train_vae_debug.remote(
            steps=steps,
            log_freq=log_freq,
            eval_freq=eval_freq,
            batch_size=batch_size,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
        )
    else:
        result = train_vae.remote(
            steps=steps,
            log_freq=log_freq,
            eval_freq=eval_freq,
            batch_size=batch_size,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
        )

    print(f"Training completed with return code: {result}")
