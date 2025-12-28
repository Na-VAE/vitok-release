#!/usr/bin/env python
"""Modal script for debugging VAE training on ImageNet-22k.

Usage:
    # Debug run with short log/eval freq
    modal run scripts/modal_train_vae.py

    # With custom settings
    modal run scripts/modal_train_vae.py --steps 1000 --log-freq 10
"""

import modal

# Create Modal app
app = modal.App("vitok-vae-train")

# Build image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
    # Install dino-perceptual from local
    .pip_install("dino-perceptual")
    # Install vitok from the repo
    .run_commands("pip install git+https://github.com/Na-VAE/vitok-release.git")
)

# Volume for checkpoints
checkpoint_volume = modal.Volume.from_name("vitok-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,  # 1 hour
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_vae_debug(
    steps: int = 500,
    log_freq: int = 10,
    eval_freq: int = 50,
    batch_size: int = 16,
    no_compile: bool = True,
):
    """Run VAE training for debugging."""
    import subprocess
    import os

    # Set HF token for ImageNet access
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Build command
    cmd = [
        "python", "-m", "scripts.train_vae",
        "--data", "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar",
        "--output_dir", "/checkpoints/vae-debug",
        "--steps", str(steps),
        "--log_freq", str(log_freq),
        "--eval_freq", str(eval_freq),
        "--batch_size", str(batch_size),
        "--save_freq", str(steps),  # Save at end only
        "--marked_freq", "0",  # No marked checkpoints
    ]

    if no_compile:
        cmd.append("--no_compile")

    print(f"Running: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


@app.local_entrypoint()
def main(
    steps: int = 500,
    log_freq: int = 10,
    eval_freq: int = 50,
    batch_size: int = 16,
    no_compile: bool = True,
):
    """Local entrypoint for Modal."""
    print(f"Starting VAE debug training...")
    print(f"  steps: {steps}")
    print(f"  log_freq: {log_freq}")
    print(f"  eval_freq: {eval_freq}")
    print(f"  batch_size: {batch_size}")
    print(f"  no_compile: {no_compile}")

    result = train_vae_debug.remote(
        steps=steps,
        log_freq=log_freq,
        eval_freq=eval_freq,
        batch_size=batch_size,
        no_compile=no_compile,
    )

    print(f"Training completed with return code: {result}")
