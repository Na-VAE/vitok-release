"""Modal wrapper for VAE evaluation.

Usage:
    # Single GPU evaluation
    modal run scripts/modal/eval_vae.py --model L-64 --max-size 512

    # With float8 quantization
    modal run scripts/modal/eval_vae.py --model 5B-64 --max-size 1024 --float8 inference

    # Full options
    modal run scripts/modal/eval_vae.py --model 5B-64 --num-samples 200 --max-size 1024 --batch-size 1
"""

import sys
from pathlib import Path

import modal

# =============================================================================
# Image & Config (defined inline to avoid import issues in remote container)
# =============================================================================

EVAL_PACKAGES = [
    "torch==2.6.0",
    "torchvision==0.21.0",
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
]

DATASET_PATHS = {
    "coco-val": "/data/coco/val2017",
    "imagenet-val": "/data/imagenet/val",
    "div8k": "/data/div8k/val",
}

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install(*EVAL_PACKAGES)
)

weights_vol = modal.Volume.from_name("vitok-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("vitok-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

EVAL_CONFIG = {
    "gpu": "H100",
    "volumes": {"/cache": weights_vol, "/data": data_vol},
    "secrets": [hf_secret],
    "timeout": 3600,
}

app = modal.App("vitok-eval")

# Build image with vitok code
image = (
    eval_image
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_file("scripts/eval_vae.py", remote_path="/root/vitok-release/scripts/eval_vae.py")
)


@app.function(image=image, **EVAL_CONFIG)
def run_eval(
    model_name: str = "L-64",
    num_samples: int = 5000,
    max_size: int = 256,
    batch_size: int = 64,
    crop_style: str = "native",
    swa_window: int | None = None,
    data_path: str | None = None,
    dataset: str | None = None,
    compile: bool = True,
    float8: str | None = None,
) -> dict:
    """Run single-GPU VAE evaluation."""
    import os
    from pathlib import Path

    sys.path.insert(0, "/root/vitok-release")
    os.environ["HF_HOME"] = "/cache/huggingface"

    # Dataset paths (inline to avoid imports)
    dataset_paths = {
        "coco-val": "/data/coco/val2017",
        "imagenet-val": "/data/imagenet/val",
        "div8k": "/data/div8k/val",
    }

    from scripts.eval_vae import evaluate

    # Resolve data path
    if data_path:
        resolved_data = data_path
    elif dataset and dataset in dataset_paths:
        resolved_data = dataset_paths[dataset]
    else:
        resolved_data = dataset_paths["coco-val"]

    # Ensure data exists
    if resolved_data.startswith("/data") and not Path(resolved_data).exists():
        if "coco" in resolved_data:
            # Download COCO
            print("Downloading COCO val2017...")
            coco_dir = Path(resolved_data)
            coco_dir.parent.mkdir(parents=True, exist_ok=True)
            zip_path = coco_dir.parent / "val2017.zip"
            os.system(f"wget -q --show-progress -O {zip_path} http://images.cocodataset.org/zips/val2017.zip")
            os.system(f"unzip -q {zip_path} -d {coco_dir.parent}")
            os.remove(zip_path)
            data_vol.commit()
        else:
            raise FileNotFoundError(f"Dataset not found: {resolved_data}")

    return evaluate(
        model_name=model_name,
        data=resolved_data,
        max_size=max_size,
        batch_size=batch_size,
        num_samples=num_samples,
        crop_style=crop_style,
        swa_window=swa_window,
        metrics=("fid", "fdd", "ssim", "psnr"),
        compile=compile,
        float8_mode=float8,
    )


@app.local_entrypoint()
def main(
    model: str = "L-64",
    num_samples: int = 5000,
    max_size: int = 256,
    batch_size: int = 64,
    crop_style: str = "native",
    swa_window: int | None = None,
    data: str | None = None,
    dataset: str | None = None,
    output_json: str | None = None,
    no_compile: bool = False,
    float8: str | None = None,
    list_models: bool = False,
    list_datasets: bool = False,
):
    """Run ViTok VAE evaluation on Modal.

    Examples:
        # Basic eval on COCO
        modal run scripts/modal/eval_vae.py --model L-64

        # 5B model at 1024p with float8
        modal run scripts/modal/eval_vae.py --model 5B-64 --max-size 1024 --batch-size 1 --float8 inference
    """
    import json

    if list_models:
        print("Available models: L-64, L-32, L-16, T-64, T-128, T-256, 5B-64")
        return

    if list_datasets:
        print("Available datasets:")
        for k, v in DATASET_PATHS.items():
            print(f"  {k}: {v}")
        return

    print(f"Model: {model}, Samples: {num_samples}, Size: {max_size}")
    print(f"Batch: {batch_size}, Compile: {not no_compile}, Float8: {float8}")
    print()

    stats = run_eval.remote(
        model_name=model,
        num_samples=num_samples,
        max_size=max_size,
        batch_size=batch_size,
        crop_style=crop_style,
        swa_window=swa_window,
        data_path=data,
        dataset=dataset,
        compile=not no_compile,
        float8=float8,
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"Results: {stats.get('model', model)}")
    print(f"{'='*50}")
    for k in ["fid", "fdd", "ssim", "psnr"]:
        if k in stats:
            print(f"  {k.upper()}: {stats[k]:.4f}")
    if "throughput_img_per_sec" in stats:
        print(f"  Throughput: {stats['throughput_img_per_sec']:.1f} img/s")
    if "avg_img_latency_ms" in stats:
        print(f"  Latency: {stats['avg_img_latency_ms']:.1f} ms/img")
    if "max_memory_allocated_gb" in stats:
        print(f"  Memory: {stats['max_memory_allocated_gb']:.2f} GB")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to: {output_json}")
