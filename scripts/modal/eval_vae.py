"""Run ViTok VAE evaluation on Modal.

This is a thin wrapper around scripts/eval_vae.py that:
1. Sets up the Modal environment with GPU and dependencies
2. Uses cached datasets from vitok-data volume (or downloads if needed)
3. Calls the same evaluate() function used locally

Usage:
    # Evaluate L-64 on COCO val2017 (default)
    modal run scripts/modal/eval_vae.py --model L-64

    # With crop style and resolution
    modal run scripts/modal/eval_vae.py --model L-64 --crop-style adm_square --max-size 256

    # Native resolution with SWA for high-res
    modal run scripts/modal/eval_vae.py --model L-64 --crop-style native --max-size 1024 --swa-window 8

    # Use cached DIV8K for high-res eval
    modal run scripts/modal/eval_vae.py --model L-64 --dataset div8k --max-size 1024

    # Quick test with fewer samples
    modal run scripts/modal/eval_vae.py --model L-64 --num-samples 100

    # Multi-GPU for large datasets (e.g., ImageNet 50K)
    modal run scripts/modal/eval_vae.py --model L-64 --dataset imagenet-val --n-gpus 8

    # Save results to JSON
    modal run scripts/modal/eval_vae.py --model L-64 --output-json results.json

    # List available models
    modal run scripts/modal/eval_vae.py --list-models

Setup:
    # Pre-download datasets to avoid re-downloading each run
    modal run scripts/modal/setup_data.py
"""

import modal
import sys

WEIGHTS_VOLUME = "vitok-weights"
DATA_VOLUME = "vitok-data"

app = modal.App("vitok-eval")

# Evaluation image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pillow>=10.0.0",
        "webdataset",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pytorch-fid>=0.3.0",
        "dino-perceptual>=0.1.0",
        "torchmetrics>=1.0.0",
        "einops",
        "torchao>=0.5.0",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_file("scripts/eval_vae.py", remote_path="/root/vitok-release/scripts/eval_vae.py")
)

# Volumes for caching
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

# Dataset presets mapping to paths on volumes
DATASET_PRESETS = {
    "coco-val": "/data/coco/val2017",
    "imagenet-val": "/data/imagenet/val",
    "div8k": "/data/div8k/val",
}


def download_coco_val(cache_dir: str) -> str:
    """Download COCO val2017 dataset if not already cached."""
    import os
    from pathlib import Path

    coco_dir = Path(cache_dir) / "coco" / "val2017"
    if coco_dir.exists() and len(list(coco_dir.glob("*.jpg"))) > 1000:
        print(f"COCO val2017 already cached: {coco_dir}")
        return str(coco_dir)

    print("Downloading COCO val2017...")
    print("TIP: Run 'modal run scripts/modal/setup_data.py' to pre-cache datasets")
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
    gpu="H100",
    volumes={"/cache": weights_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def run_eval(
    model_name: str = "L-64",
    num_samples: int = 5000,
    max_size: int = 256,
    batch_size: int = 64,
    crop_style: str = "native",
    swa_window: int | None = None,
    data_path: str | None = None,
    dataset: str | None = None,
    save_visuals: int = 0,
    compile: bool = True,
    float8: bool = False,
) -> dict:
    """Run VAE evaluation using the shared evaluate() function.

    Args:
        model_name: Pretrained model name (e.g., "L-64", "L-16")
        num_samples: Number of samples to evaluate
        max_size: Maximum image size
        batch_size: Batch size for evaluation
        crop_style: Crop style (adm_square, native)
        swa_window: Sliding window attention radius (None=full attention)
        data_path: Optional custom data path
        dataset: Dataset preset (coco-val, div8k, imagenet-val)
        save_visuals: Number of sample images to save (0=none)

    Returns:
        Dictionary with evaluation metrics (includes 'visuals' key with base64 encoded images if save_visuals > 0)
    """
    import os
    import base64
    from pathlib import Path

    # Add vitok to path
    sys.path.insert(0, "/root/vitok-release")

    # Import the shared evaluate function
    from scripts.eval_vae import evaluate

    # Set HF cache
    os.environ["HF_HOME"] = "/cache/huggingface"

    # Resolve data path
    if data_path:
        resolved_data = data_path
    elif dataset:
        if dataset in DATASET_PRESETS:
            resolved_data = DATASET_PRESETS[dataset]
            # Check if local dataset exists
            if resolved_data.startswith("/data"):
                if not Path(resolved_data).exists():
                    if dataset == "coco-val":
                        print(f"Dataset not cached at {resolved_data}, downloading...")
                        resolved_data = download_coco_val("/data")
                        data_vol.commit()
                    elif dataset == "imagenet-val":
                        raise FileNotFoundError(
                            f"ImageNet-1k val not cached at {resolved_data}. "
                            f"Run 'modal run scripts/modal/setup_data.py --dataset imagenet' first. "
                            f"Note: Requires HF token with accepted license for ILSVRC/imagenet-1k."
                        )
                    else:
                        raise FileNotFoundError(
                            f"Dataset {dataset} not found at {resolved_data}. "
                            f"Run 'modal run scripts/modal/setup_data.py --dataset {dataset.split('-')[0]}' first."
                        )
        else:
            raise ValueError(f"Unknown dataset preset: {dataset}. Available: {list(DATASET_PRESETS.keys())}")
    else:
        # Default to COCO
        resolved_data = DATASET_PRESETS["coco-val"]
        if not Path(resolved_data).exists():
            resolved_data = download_coco_val("/data")
            data_vol.commit()

    print(f"Using data: {resolved_data}")

    # Setup output directory for visuals
    output_dir = Path("/tmp/eval_output") if save_visuals > 0 else None

    # Run evaluation using shared function
    stats = evaluate(
        model_name=model_name,
        data=resolved_data,
        max_size=max_size,
        batch_size=batch_size,
        num_samples=num_samples,
        crop_style=crop_style,
        swa_window=swa_window,
        metrics=("fid", "fdd", "ssim", "psnr"),
        compile=compile,
        float8_mode="inference" if float8 else None,
        verbose=True,
        save_visuals=save_visuals,
        output_dir=output_dir,
    )

    # Read and encode visuals if saved
    if save_visuals > 0 and output_dir and output_dir.exists():
        visuals = {}
        grid_path = output_dir / "comparison_grid.jpg"
        if grid_path.exists():
            with open(grid_path, "rb") as f:
                visuals["comparison_grid"] = base64.b64encode(f.read()).decode()
        stats["visuals"] = visuals

    return stats


@app.function(
    image=image,
    gpu="A100:8",
    volumes={"/cache": weights_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def run_eval_multi_gpu(
    model_name: str = "L-64",
    num_samples: int = 50000,
    max_size: int = 256,
    batch_size: int = 64,
    crop_style: str = "native",
    swa_window: int | None = None,
    data_path: str | None = None,
    dataset: str | None = None,
    save_visuals: int = 0,
    n_gpus: int = 8,
) -> dict:
    """Run multi-GPU VAE evaluation using torchrun.

    Uses distributed data parallel to split evaluation across multiple GPUs.
    """
    import os
    import subprocess
    import json
    import base64
    from pathlib import Path

    # Set environment
    os.environ["HF_HOME"] = "/cache/huggingface"

    # Resolve data path
    if data_path:
        resolved_data = data_path
    elif dataset:
        if dataset in DATASET_PRESETS:
            resolved_data = DATASET_PRESETS[dataset]
            if resolved_data.startswith("/data"):
                if not Path(resolved_data).exists():
                    if dataset == "coco-val":
                        print(f"Dataset not cached at {resolved_data}, downloading...")
                        resolved_data = download_coco_val("/data")
                        data_vol.commit()
                    elif dataset == "imagenet-val":
                        raise FileNotFoundError(
                            f"ImageNet-1k val not cached at {resolved_data}. "
                            f"Run 'modal run scripts/modal/setup_data.py --dataset imagenet' first."
                        )
                    else:
                        raise FileNotFoundError(
                            f"Dataset {dataset} not found at {resolved_data}. "
                            f"Run 'modal run scripts/modal/setup_data.py --dataset {dataset.split('-')[0]}' first."
                        )
        else:
            raise ValueError(f"Unknown dataset preset: {dataset}")
    else:
        resolved_data = DATASET_PRESETS["coco-val"]
        if not Path(resolved_data).exists():
            resolved_data = download_coco_val("/data")
            data_vol.commit()

    print(f"Using data: {resolved_data}")
    print(f"Running with {n_gpus} GPUs")

    # Setup output directory
    output_dir = Path("/tmp/eval_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    # Build torchrun command
    # Note: eval_vae.py automatically disables compile when distributed is detected
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        "/root/vitok-release/scripts/eval_vae.py",
        "--model", model_name,
        "--data", resolved_data,
        "--max-size", str(max_size),
        "--batch-size", str(batch_size),
        "--num-samples", str(num_samples),
        "--crop-style", crop_style,
        "--output-json", str(results_file),
    ]
    if swa_window is not None:
        cmd.extend(["--swa-window", str(swa_window)])
    if save_visuals > 0:
        cmd.extend(["--save-visuals", str(save_visuals), "--output-dir", str(output_dir)])

    print(f"Running: {' '.join(cmd)}")

    # Run evaluation
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/vitok-release"
    result = subprocess.run(cmd, env=env, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")

    # Load results
    if results_file.exists():
        with open(results_file) as f:
            stats = json.load(f)
    else:
        stats = {"error": "Results file not found"}

    # Encode visuals if saved
    if save_visuals > 0:
        visuals = {}
        grid_path = output_dir / "comparison_grid.jpg"
        if grid_path.exists():
            with open(grid_path, "rb") as f:
                visuals["comparison_grid"] = base64.b64encode(f.read()).decode()
        stats["visuals"] = visuals

    return stats


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
    output_dir: str | None = None,
    save_visuals: int = 8,
    n_gpus: int = 1,
    no_compile: bool = False,
    float8: bool = False,
    list_models: bool = False,
    list_datasets: bool = False,
):
    """Run ViTok VAE evaluation on Modal.

    Args:
        model: Pretrained model name (default: L-64)
        num_samples: Number of samples to evaluate (default: 5000)
        max_size: Maximum image size (default: 512)
        batch_size: Batch size (default: 16)
        crop_style: Crop style - adm_square or native (default: native)
        swa_window: Sliding window attention radius (default: None)
        data: Custom data path (default: COCO val2017)
        dataset: Dataset preset (coco-val, div8k, imagenet-val)
        output_json: Save results to JSON file
        output_dir: Directory to save visuals (default: results/<model>_<dataset>_<size>)
        save_visuals: Number of sample images to save (default: 8, 0=none)
        n_gpus: Number of GPUs (1=single A100, 8=8xA100 with torchrun)
        list_models: List available pretrained models and exit
        list_datasets: List available dataset presets and exit
    """
    import json
    import base64
    from pathlib import Path

    if list_models:
        aliases = {
            "L-64": "Ld4-Ld24/1x16x64 (philippehansen/ViTok-L-16x64)",
            "L-32": "Ld4-Ld24/1x32x64 (Na-VAE/ViTok-L-32)",
            "L-16": "Ld4-Ld24/1x16x16 (Na-VAE/ViTok-L-16)",
            "T-32x64": "Td2-Td12/1x32x64 (philippehansen/ViTok-T-32x64)",
        }
        print("Available pretrained models:")
        print()
        for alias, full_name in sorted(aliases.items()):
            print(f"  {alias:10s} -> {full_name}")
        return

    if list_datasets:
        print("Available dataset presets:")
        print()
        print("  coco-val     COCO val2017 (5K images, general eval)")
        print("  imagenet-val ImageNet-1k val (50K images, requires HF auth)")
        print("  div8k        DIV8K validation (high-res, 1024p+)")
        print()
        print("Setup:")
        print("  modal run scripts/modal/setup_data.py                    # COCO only")
        print("  modal run scripts/modal/setup_data.py --dataset imagenet # ImageNet")
        print("  modal run scripts/modal/setup_data.py --dataset div8k    # DIV8K")
        return

    print(f"Model: {model}")
    print(f"Samples: {num_samples}")
    print(f"Max size: {max_size}")
    print(f"Batch size: {batch_size}")
    print(f"Crop style: {crop_style}")
    print(f"SWA window: {swa_window}")
    print(f"Save visuals: {save_visuals}")
    print(f"GPUs: {n_gpus}")
    print(f"Compile: {not no_compile}")
    print(f"Float8: {float8}")
    if dataset:
        print(f"Dataset: {dataset}")
    elif data:
        print(f"Data: {data}")
    else:
        print("Data: COCO val2017 (from vitok-data volume)")
    print(f"\nRunning evaluation on Modal...\n")

    if n_gpus > 1:
        # Multi-GPU with torchrun
        stats = run_eval_multi_gpu.remote(
            model_name=model,
            num_samples=num_samples,
            max_size=max_size,
            batch_size=batch_size,
            crop_style=crop_style,
            swa_window=swa_window,
            data_path=data,
            dataset=dataset,
            save_visuals=save_visuals,
            n_gpus=n_gpus,
        )
    else:
        # Single GPU
        stats = run_eval.remote(
            model_name=model,
            num_samples=num_samples,
            max_size=max_size,
            batch_size=batch_size,
            crop_style=crop_style,
            swa_window=swa_window,
            data_path=data,
            dataset=dataset,
            save_visuals=save_visuals,
            compile=not no_compile,
            float8=float8,
        )

    print(f"\n{'='*50}")
    print(f"Evaluation Results: {stats.get('model', model)}")
    print(f"{'='*50}")
    print(f"Variant: {stats.get('variant', 'N/A')}")
    print(f"Samples: {stats.get('samples', 'N/A')}")
    print(f"Crop style: {stats.get('crop_style', 'N/A')}")
    print(f"SWA window: {stats.get('swa_window', 'N/A')}")
    if "throughput_img_per_sec" in stats:
        print(f"Throughput: {stats['throughput_img_per_sec']:.1f} img/s")
    print()
    if "total_gflops" in stats:
        print("FLOPs:")
        print(f"  Total: {stats['total_gflops']:.2f} GFLOPs")
        print()
    print("Metrics:")
    for k in ["fid", "fdd", "ssim", "psnr"]:
        if k in stats:
            print(f"  {k.upper():6s}: {stats[k]:.4f}")

    # Save visuals locally
    if "visuals" in stats and stats["visuals"]:
        # Determine output directory
        if output_dir:
            vis_dir = Path(output_dir)
        else:
            ds_name = dataset or "coco-val"
            vis_dir = Path(f"results/{model}_{ds_name}_{max_size}p_{crop_style}")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison grid
        if "comparison_grid" in stats["visuals"]:
            grid_path = vis_dir / "comparison_grid.jpg"
            with open(grid_path, "wb") as f:
                f.write(base64.b64decode(stats["visuals"]["comparison_grid"]))
            print(f"\nSaved comparison grid to: {grid_path}")

        # Remove visuals from stats before saving JSON
        del stats["visuals"]

    if output_json:
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {output_json}")
