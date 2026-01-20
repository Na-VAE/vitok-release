"""Modal environment configuration for vitok-release GPU tests.

This module defines the Modal image and app configuration used by all GPU tests.

Usage:
    from tests.gpu.env import app, image, VITOK_PATH, V2_PATH
"""

import modal

# Paths inside Modal container
VITOK_PATH = "/root/vitok-release"
V2_PATH = "/root/vitokv2"

# Base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
        "pytest>=7.0.0",
        "requests",
        "diffusers>=0.31.0",
        "ml_collections",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
    .add_local_dir("tests", remote_path=f"{VITOK_PATH}/tests")
)

# Image with both vitok-release and vitokv2 for compatibility testing
compat_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
        "pytest>=7.0.0",
        "requests",
        "diffusers>=0.31.0",
        "ml_collections",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
    .add_local_dir("../vitokv2/vitok", remote_path=f"{V2_PATH}/vitok")
)

# Modal app for all tests
app = modal.App("vitok-release-tests")

__all__ = ["app", "image", "compat_image", "VITOK_PATH", "V2_PATH"]
