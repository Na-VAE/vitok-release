"""Run ViTok VAE evaluation on Modal.

This is a thin wrapper around scripts/eval_vae.py that:
1. Sets up the Modal environment with GPU and dependencies
2. Downloads COCO val2017 if needed
3. Calls the same evaluate() function used locally

Usage:
    # Evaluate L-64 on COCO val2017 (default)
    modal run scripts/modal/eval_vae.py

    # Evaluate specific model
    modal run scripts/modal/eval_vae.py --model L-16

    # Quick test with fewer samples
    modal run scripts/modal/eval_vae.py --model L-64 --num-samples 100

    # Full evaluation
    modal run scripts/modal/eval_vae.py --model L-64 --num-samples 5000

    # List available models
    modal run scripts/modal/eval_vae.py --list-models
"""

import modal
import sys

VOLUME_NAME = "vitok-weights"

app = modal.App("vitok-eval")

# Evaluation image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pillow>=10.0.0",
        "webdataset",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pytorch-fid>=0.3.0",
        "dino-perceptual>=0.1.0",
        "torchmetrics>=1.0.0",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_file("scripts/eval_vae.py", remote_path="/root/vitok-release/scripts/eval_vae.py")
)

# Volume for caching weights and datasets
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def download_coco_val(cache_dir: str) -> str:
    """Download COCO val2017 dataset if not already cached."""
    import os
    from pathlib import Path

    coco_dir = Path(cache_dir) / "coco" / "val2017"
    if coco_dir.exists() and len(list(coco_dir.glob("*.jpg"))) > 1000:
        print(f"COCO val2017 already cached: {coco_dir}")
        return str(coco_dir)

    print("Downloading COCO val2017...")
    coco_dir.parent.mkdir(parents=True, exist_ok=True)

    zip_path = coco_dir.parent / "val2017.zip"
    os.system(f"wget -q --show-progress -O {zip_path} http://images.cocodataset.org/zips/val2017.zip")
    os.system(f"unzip -q {zip_path} -d {coco_dir.parent}")
    os.remove(zip_path)

    n_images = len(list(coco_dir.glob("*.jpg")))
    print(f"Downloaded {n_images} images to {coco_dir}")
    return str(coco_dir)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def run_eval(
    model_name: str = "L-64",
    num_samples: int = 5000,
    max_size: int = 512,
    batch_size: int = 16,
    data_path: str | None = None,
) -> dict:
    """Run VAE evaluation using the shared evaluate() function.

    Args:
        model_name: Pretrained model name (e.g., "L-64", "L-16")
        num_samples: Number of samples to evaluate
        max_size: Maximum image size
        batch_size: Batch size for evaluation
        data_path: Optional custom data path (default: COCO val2017)

    Returns:
        Dictionary with evaluation metrics
    """
    import os

    # Add vitok to path
    sys.path.insert(0, "/root/vitok-release")

    # Import the shared evaluate function
    from scripts.eval_vae import evaluate

    # Set HF cache
    os.environ["HF_HOME"] = "/cache/huggingface"

    # Download COCO if no custom data path
    if data_path is None:
        data_path = download_coco_val("/cache/datasets")
        vol.commit()

    # Run evaluation using shared function
    stats = evaluate(
        model_name=model_name,
        data=data_path,
        max_size=max_size,
        batch_size=batch_size,
        num_samples=num_samples,
        metrics=("fid", "fdd", "ssim", "psnr"),
        compile=True,
        verbose=True,
    )

    return stats


@app.local_entrypoint()
def main(
    model: str = "L-64",
    num_samples: int = 5000,
    max_size: int = 512,
    batch_size: int = 16,
    data: str | None = None,
    list_models: bool = False,
):
    """Run ViTok VAE evaluation on Modal.

    Args:
        model: Pretrained model name (default: L-64)
        num_samples: Number of samples to evaluate (default: 5000)
        max_size: Maximum image size (default: 512)
        batch_size: Batch size (default: 16)
        data: Custom data path (default: COCO val2017)
        list_models: List available pretrained models and exit
    """
    if list_models:
        aliases = {
            "L-64": "Ld4-Ld24/1x16x64",
            "L-32": "Ld4-Ld24/1x32x64",
            "L-16": "Ld4-Ld24/1x16x16",
            "T-64": "Td2-Td12/1x16x64",
            "T-128": "Td2-Td12/1x16x128",
            "T-256": "Td2-Td12/1x16x256",
        }
        print("Available pretrained models:")
        print()
        for alias, full_name in sorted(aliases.items()):
            print(f"  {alias:10s} -> {full_name}")
        return

    print(f"Model: {model}")
    print(f"Samples: {num_samples}")
    print(f"Max size: {max_size}")
    print(f"Batch size: {batch_size}")
    if data:
        print(f"Data: {data}")
    else:
        print("Data: COCO val2017 (will download if needed)")
    print(f"\nRunning evaluation on Modal...\n")

    stats = run_eval.remote(model, num_samples, max_size, batch_size, data)

    print(f"\n{'='*50}")
    print(f"Evaluation Results: {stats.get('model', model)}")
    print(f"{'='*50}")
    print(f"Variant: {stats.get('variant', 'N/A')}")
    print(f"Samples: {stats.get('samples', 'N/A')}")
    print()
    print("Metrics:")
    for k in ["fid", "fdd", "ssim", "psnr"]:
        if k in stats:
            print(f"  {k.upper():6s}: {stats[k]:.4f}")
