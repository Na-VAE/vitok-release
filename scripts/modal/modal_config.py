"""Shared Modal configuration for ViTok scripts.

Usage:
    from scripts.modal.modal_config import image, gpu, DATASET_PATHS

    app = modal.App("my-app")

    @app.function(image=image, **gpu("H100"))
    def run_eval(...):
        ...

How it works:
    - `image`: Pre-built image with vitok code, torch, and all dependencies
    - `gpu(name, timeout)`: Returns config dict with GPU, volumes, secrets
    - Scripts use `modal run script.py` which finds the app and runs it
"""

import modal

# =============================================================================
# Image with all dependencies
# =============================================================================

PACKAGES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "safetensors>=0.4.0",
    "huggingface_hub>=0.23.0",
    "pillow>=10.0.0",
    "webdataset",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "einops",
    "pytorch-fid>=0.3.0",
    "dino-perceptual>=0.1.0",
    "torchmetrics>=1.0.0",
    "torchao>=0.5.0",
    "wandb",
    # For baseline VAEs and streaming datasets
    "diffusers>=0.25.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "datasets>=2.16.0",
    # Note: flash-attn removed - using flex_attention for 2D SWA instead
    # flash-attn requires packaging + complex build, not worth the hassle
]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "unzip", "curl")
    .pip_install(*PACKAGES)
    .env({
        "PYTHONPATH": "/root/vitok-release",
        # Use weights from volume (run setup_weights.py first)
        "HF_HOME": "/data/weights/huggingface",
        "TORCH_HOME": "/data/weights/torch",
    })
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_dir("scripts", remote_path="/root/vitok-release/scripts")
)


# =============================================================================
# Volumes & Secrets
# =============================================================================

data_vol = modal.Volume.from_name("vitok-data", create_if_missing=True)
output_vol = modal.Volume.from_name("vitok-output", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# =============================================================================
# GPU Config
# =============================================================================

def gpu(name: str, timeout: int = 3600) -> dict:
    """Config for a GPU. Usage: @app.function(image=image, **gpu("H100"))"""
    return {
        "gpu": name,
        "volumes": {"/data": data_vol, "/output": output_vol},
        "secrets": [hf_secret],
        "timeout": timeout,
    }


# =============================================================================
# Dataset Paths (on data_vol)
# =============================================================================

DATASET_PATHS = {
    "coco-val": "/data/coco/val2017",
    "imagenet-val": "/data/imagenet/val",
    "div8k": "/data/div8k/train",  # 1500 images from DIV8K train set (Iceclear/DIV8K_TrainingSet)
    # Benchmark datasets for visual comparison
    "kodak": "/data/benchmarks/kodak",  # 24 images, 768x512
    "set14": "/data/benchmarks/set14",  # 14 images, classic SR benchmark
    "urban100": "/data/benchmarks/urban100",  # 100 images, architecture
    "bsd100": "/data/benchmarks/bsd100",  # 100 images, natural scenes
    "celeba": "/data/benchmarks/celeba",  # 50 images, faces
    "challenge": "/data/benchmarks/challenge",  # 5 images, hand-picked hard cases
}
