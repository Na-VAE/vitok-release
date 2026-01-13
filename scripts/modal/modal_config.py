"""Shared Modal configuration for all ViTok scripts.

This module defines reusable Modal resources (images, volumes, secrets)
and configuration presets that can be unpacked into @app.function(**CONFIG).

Usage in Modal scripts:
    from modal_config import EVAL_CONFIG, base_image, with_vitok_code

    app = modal.App("vitok-eval")
    image = with_vitok_code()

    @app.function(image=image, **EVAL_CONFIG)
    def run_eval(...):
        ...

Usage with --modal flag in main scripts:
    from modal_config import run_on_modal, EVAL_CONFIG

    if args.modal:
        result = run_on_modal(
            fn_path="scripts.eval_vae:evaluate",
            config=EVAL_CONFIG,
            kwargs={...}
        )
"""

import modal

# =============================================================================
# Base Image
# =============================================================================

BASE_PACKAGES = [
    "torch==2.6.0",
    "torchvision==0.21.0",
    "safetensors>=0.4.0",
    "huggingface_hub>=0.23.0",
    "pillow>=10.0.0",
    "webdataset",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "einops",
]

EVAL_PACKAGES = BASE_PACKAGES + [
    "pytorch-fid>=0.3.0",
    "dino-perceptual>=0.1.0",
    "torchmetrics>=1.0.0",
    "torchao>=0.5.0",
]

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install(*BASE_PACKAGES)
)

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install(*EVAL_PACKAGES)
)


def with_vitok_code(image: modal.Image = None, scripts: list[str] = None):
    """Add vitok package and optional scripts to an image.

    Args:
        image: Base image (default: eval_image)
        scripts: List of script paths relative to repo root (e.g., ["scripts/eval_vae.py"])

    Returns:
        Image with vitok code added
    """
    if image is None:
        image = eval_image

    image = image.add_local_dir("vitok", remote_path="/root/vitok-release/vitok")

    if scripts:
        for script in scripts:
            remote_path = f"/root/vitok-release/{script}"
            image = image.add_local_file(script, remote_path=remote_path)

    return image


# =============================================================================
# Volumes & Secrets
# =============================================================================

weights_vol = modal.Volume.from_name("vitok-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("vitok-data", create_if_missing=True)
results_vol = modal.Volume.from_name("vitok-eval-results", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret")


# =============================================================================
# Function Configs (unpack with @app.function(**CONFIG))
# =============================================================================

# Inference: cheap GPU, just weights
INFERENCE_CONFIG = {
    "gpu": "T4",
    "volumes": {"/cache": weights_vol},
    "secrets": [hf_secret],
    "timeout": 300,
}

# Evaluation: fast GPU, weights + data
EVAL_CONFIG = {
    "gpu": "H100",
    "volumes": {"/cache": weights_vol, "/data": data_vol},
    "secrets": [hf_secret],
    "timeout": 3600,
}

# Multi-GPU evaluation: 8x A100
EVAL_MULTI_GPU_CONFIG = {
    "gpu": "A100:8",
    "volumes": {"/cache": weights_vol, "/data": data_vol},
    "secrets": [hf_secret],
    "timeout": 7200,
}

# Batch evaluation: 8x A100 + results volume
BATCH_EVAL_CONFIG = {
    "gpu": "A100:8",
    "volumes": {"/cache": weights_vol, "/data": data_vol, "/results": results_vol},
    "secrets": [hf_secret],
    "timeout": 43200,  # 12 hours
}

# Training: 8x A100
TRAINING_CONFIG = {
    "gpu": "A100:8",
    "volumes": {"/cache": weights_vol, "/data": data_vol},
    "secrets": [hf_secret],
    "timeout": 86400,  # 24 hours
}


# =============================================================================
# Dataset Paths (on data_vol)
# =============================================================================

DATASET_PATHS = {
    "coco-val": "/data/coco/val2017",
    "imagenet-val": "/data/imagenet/val",
    "div8k": "/data/div8k/val",
}


# =============================================================================
# Helper: Run function on Modal
# =============================================================================

def multi_gpu_modal(
    app_name: str = "vitok",
    gpu: str = "H100",
    timeout: int = 3600,
    volumes: dict = None,
    secrets: list = None,
    image: modal.Image = None,
):
    """Factory to create Modal-wrapped functions for multi-GPU execution.

    Usage:
        @multi_gpu_modal("vitok-eval", gpu="H100", timeout=3600)
        def run():
            return evaluate(**kwargs)
        stats = run()

    Args:
        app_name: Name for the Modal app
        gpu: GPU spec (e.g., "H100", "A100:8")
        timeout: Function timeout in seconds
        volumes: Volume mappings (default: weights + data volumes)
        secrets: Secrets list (default: HF secret)
        image: Modal image (default: eval_image with vitok code)

    Returns:
        Decorator that wraps function for Modal execution
    """
    def decorator(fn):
        # Build image
        _image = image if image is not None else with_vitok_code(eval_image)

        # Default volumes/secrets
        _volumes = volumes if volumes is not None else {"/cache": weights_vol, "/data": data_vol}
        _secrets = secrets if secrets is not None else [hf_secret]

        # Create app
        app = modal.App(app_name)

        @app.function(image=_image, gpu=gpu, timeout=timeout, volumes=_volumes, secrets=_secrets)
        def _remote():
            import sys
            import os as _os

            sys.path.insert(0, "/root/vitok-release")
            _os.environ["HF_HOME"] = "/cache/huggingface"
            return fn()

        # Run and return result
        with app.run():
            return _remote.remote()

    return decorator
