"""Pre-build Modal environment for ViTok inference.

This script builds and caches the Modal image with all required dependencies.
Run this once to speed up subsequent inference runs.

Usage:
    modal run scripts/modal/setup_env.py
"""

import modal

app = modal.App("vitok-setup-env")

# Inference image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pillow>=10.0.0",
        "scikit-image",
        "numpy>=1.24.0",
    )
)


@app.function(image=image, gpu="T4")
def check_environment():
    """Verify the environment is set up correctly."""
    import torch
    import safetensors
    import huggingface_hub
    from skimage import data

    print("=" * 50)
    print("ViTok Modal Environment")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"safetensors version: {safetensors.__version__}")
    print(f"huggingface_hub version: {huggingface_hub.__version__}")

    # Test astronaut image loads
    astronaut = data.astronaut()
    print(f"Test image (astronaut): {astronaut.shape}")

    print("=" * 50)
    print("Environment ready!")
    print("=" * 50)


@app.local_entrypoint()
def main():
    """Build the Modal image and verify environment."""
    print("Building Modal image with inference dependencies...")
    print("This may take a few minutes on first run.\n")
    check_environment.remote()
    print("\nModal environment is ready for inference!")
    print("You can now run: modal run scripts/modal/inference.py")
