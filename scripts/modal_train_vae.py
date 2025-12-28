#!/usr/bin/env python
"""Modal script for debugging VAE training on ImageNet-22k.

Usage:
    # Debug run (500 steps, ~$3-4 on A100)
    modal run scripts/modal_train_vae.py

    # Custom settings
    modal run scripts/modal_train_vae.py --steps 1000 --log-freq 50 --eval-freq 200

    # With wandb logging
    modal run scripts/modal_train_vae.py --wandb-project vitok-debug
"""

import modal

# Create Modal app
app = modal.App("vitok-vae-train")

# Build image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "numpy",
        "tqdm",
        "wandb",
        "safetensors",
        "webdataset",
        "torchmetrics",
        "Pillow",
        "huggingface_hub",
    )
    # Install dino-perceptual
    .pip_install("dino-perceptual")
    # Install vitok from the repo
    .run_commands("pip install git+https://github.com/Na-VAE/vitok-release.git")
)

# Volume for checkpoints
checkpoint_volume = modal.Volume.from_name("vitok-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_vae_debug(
    steps: int = 500,
    log_freq: int = 10,
    eval_freq: int = 100,
    batch_size: int = 32,
    wandb_project: str = None,
    wandb_name: str = None,
):
    """Run VAE training for debugging on ImageNet-22k with ImageNet-1k val."""
    import subprocess
    import os

    # Set HF token for ImageNet access
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print(f"HF token set (length: {len(hf_token)})")

    # Build command
    cmd = [
        "python", "-m", "scripts.train_vae",
        # Data
        "--data", "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar",
        "--eval_data", "hf://ILSVRC/imagenet-1k/val/*.tar",
        # Training params
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--max_size", "512",
        "--max_tokens", "256",
        # Logging
        "--log_freq", str(log_freq),
        "--eval_freq", str(eval_freq),
        "--output_dir", "/checkpoints/vae-debug",
        "--save_freq", str(max(steps, 500)),  # Save at end
        "--marked_freq", "0",  # No marked checkpoints for debug
        # System
        "--no_compile",  # Faster startup for debug
    ]

    if wandb_project:
        cmd.extend(["--wandb_project", wandb_project])
    if wandb_name:
        cmd.extend(["--wandb_name", wandb_name])

    print(f"Running command:")
    print(f"  {' '.join(cmd)}")
    print()

    # Run training
    result = subprocess.run(cmd)
    return result.returncode


@app.local_entrypoint()
def main(
    steps: int = 500,
    log_freq: int = 10,
    eval_freq: int = 100,
    batch_size: int = 32,
    wandb_project: str = None,
    wandb_name: str = None,
):
    """Local entrypoint for Modal.

    Args:
        steps: Number of training steps (default: 500)
        log_freq: Logging frequency (default: 10)
        eval_freq: Evaluation frequency (default: 100)
        batch_size: Batch size per GPU (default: 32)
        wandb_project: Optional wandb project name
        wandb_name: Optional wandb run name
    """
    print("=" * 60)
    print("ViTok VAE Debug Training on Modal")
    print("=" * 60)
    print(f"  Model: Ld2-Ld22/1x16x64")
    print(f"  GPU: A100-80GB")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Log freq: {log_freq}")
    print(f"  Eval freq: {eval_freq}")
    print(f"  Train data: ImageNet-22k (first 50 shards)")
    print(f"  Eval data: ImageNet-1k val")
    if wandb_project:
        print(f"  WandB: {wandb_project}/{wandb_name or 'auto'}")
    print("=" * 60)
    print()

    # Estimate cost
    est_time_min = steps * 3 / 60  # ~3 sec/step estimate
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
    )

    print(f"Training completed with return code: {result}")
