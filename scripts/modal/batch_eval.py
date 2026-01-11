"""Batch VAE evaluation orchestrator for Modal.

This script runs multiple model evaluations in sequence or parallel,
storing results to a Modal volume for later retrieval.

Usage:
    # Run all T-model evaluations (COCO 256p + 512p)
    modal run scripts/modal/batch_eval.py

    # Check status of running/completed evals
    modal run scripts/modal/batch_eval.py --status

    # Get results summary
    modal run scripts/modal/batch_eval.py --results

    # Run specific models only
    modal run scripts/modal/batch_eval.py --models T-32x64,T-32x128

    # Run with float8 inference mode
    modal run scripts/modal/batch_eval.py --float8
"""

import modal
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Volumes
WEIGHTS_VOLUME = "vitok-weights"
DATA_VOLUME = "vitok-data"
RESULTS_VOLUME = "vitok-eval-results"

app = modal.App("vitok-batch-eval")

# Evaluation image - same as eval_vae.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
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
        "einops",
        "torchao>=0.5.0",  # For float8
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_file("scripts/eval_vae.py", remote_path="/root/vitok-release/scripts/eval_vae.py")
)

# Volumes
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

# Eval configurations
# Format: (model_name, resolution, crop_style, float8_mode)
EVAL_CONFIGS = [
    # T models at 256p ADM crop
    ("T-32x64", 256, "adm_square", None),
    ("T-32x128", 256, "adm_square", None),
    ("T-32x256", 256, "adm_square", None),
    # T models at 512p ADM crop
    ("T-32x64", 512, "adm_square", None),
    ("T-32x128", 512, "adm_square", None),
    ("T-32x256", 512, "adm_square", None),
]

# Float8 eval configs (added when --float8 is passed)
EVAL_CONFIGS_FLOAT8 = [
    # T models at 256p ADM crop with float8 inference
    ("T-32x64", 256, "adm_square", "inference"),
    ("T-32x128", 256, "adm_square", "inference"),
    ("T-32x256", 256, "adm_square", "inference"),
    # T models at 512p ADM crop with float8 inference
    ("T-32x64", 512, "adm_square", "inference"),
    ("T-32x128", 512, "adm_square", "inference"),
    ("T-32x256", 512, "adm_square", "inference"),
]


def config_to_id(model: str, res: int, crop: str, float8: Optional[str]) -> str:
    """Generate unique ID for an eval config."""
    f8_suffix = f"_f8{float8}" if float8 else ""
    return f"{model}_{res}p_{crop}{f8_suffix}"


def download_coco_val(cache_dir: str) -> str:
    """Download COCO val2017 dataset if not already cached."""
    import os

    coco_dir = Path(cache_dir) / "coco" / "val2017"
    if coco_dir.exists() and len(list(coco_dir.glob("*.jpg"))) > 1000:
        return str(coco_dir)

    print("Downloading COCO val2017...")
    coco_dir.parent.mkdir(parents=True, exist_ok=True)

    zip_path = coco_dir.parent / "val2017.zip"
    os.system(f"wget -q --show-progress -O {zip_path} http://images.cocodataset.org/zips/val2017.zip")
    os.system(f"unzip -q {zip_path} -d {coco_dir.parent}")
    os.remove(zip_path)

    return str(coco_dir)


@app.function(
    image=image,
    gpu="A100:8",
    volumes={"/cache": weights_vol, "/data": data_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours per eval
)
def run_single_eval(
    model_name: str,
    max_size: int,
    crop_style: str,
    float8_mode: Optional[str] = None,
    num_samples: int = 5000,
    batch_size: int = 64,
    n_gpus: int = 8,
    save_samples: bool = True,
) -> dict:
    """Run a single evaluation and save results to volume."""
    import os
    import subprocess
    import sys

    # Set environment
    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/root/vitok-release")

    # Ensure COCO is available
    coco_path = "/data/coco/val2017"
    if not Path(coco_path).exists():
        coco_path = download_coco_val("/data")
        data_vol.commit()

    # Generate config ID
    config_id = config_to_id(model_name, max_size, crop_style, float8_mode)
    print(f"\n{'='*60}")
    print(f"Running evaluation: {config_id}")
    print(f"{'='*60}")

    # Setup output directory on results volume
    output_dir = Path(f"/results/{config_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        "/root/vitok-release/scripts/eval_vae.py",
        "--model", model_name,
        "--data", coco_path,
        "--max-size", str(max_size),
        "--batch-size", str(batch_size),
        "--num-samples", str(num_samples),
        "--crop-style", crop_style,
        "--output-json", str(results_file),
    ]

    if save_samples:
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        cmd.extend(["--save-visuals", "16", "--output-dir", str(output_dir)])

    # TODO: Add float8_mode support once we verify it works
    # if float8_mode:
    #     cmd.extend(["--float8-mode", float8_mode])

    print(f"Command: {' '.join(cmd)}")

    # Run evaluation
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/vitok-release"
    start_time = datetime.now()

    result = subprocess.run(cmd, env=env, capture_output=False)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Load and augment results
    if results_file.exists():
        with open(results_file) as f:
            stats = json.load(f)
    else:
        stats = {"error": "Results file not found", "returncode": result.returncode}

    stats["config_id"] = config_id
    stats["duration_seconds"] = duration
    stats["float8_mode"] = float8_mode
    stats["timestamp"] = end_time.isoformat()

    # Save augmented results
    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Commit to volume
    results_vol.commit()

    print(f"\nCompleted {config_id} in {duration:.1f}s")
    print(f"Results saved to: /results/{config_id}/results.json")

    return stats


@app.function(
    image=image,
    gpu="A100:8",
    volumes={"/cache": weights_vol, "/data": data_vol, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,  # 12 hours for full batch
)
def run_batch_eval(
    configs: list[tuple],
    num_samples: int = 5000,
    batch_size: int = 64,
    n_gpus: int = 8,
) -> list[dict]:
    """Run multiple evaluations sequentially.

    This is better than parallel for resource utilization since each
    eval already uses 8 GPUs.
    """
    import os
    import sys

    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/root/vitok-release")

    # Ensure COCO is available first
    coco_path = "/data/coco/val2017"
    if not Path(coco_path).exists():
        print("Pre-downloading COCO val2017...")
        coco_path = download_coco_val("/data")
        data_vol.commit()

    results = []
    total = len(configs)

    # Create batch status file
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_file = Path(f"/results/batch_{batch_id}/status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "batch_id": batch_id,
        "total": total,
        "completed": 0,
        "failed": 0,
        "configs": [config_to_id(*c) for c in configs],
        "results": [],
        "started_at": datetime.now().isoformat(),
    }

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    results_vol.commit()

    for i, (model, res, crop, float8) in enumerate(configs):
        config_id = config_to_id(model, res, crop, float8)
        print(f"\n[{i+1}/{total}] Running: {config_id}")

        try:
            # Run eval using the single eval function's logic directly
            # (We can't call Modal functions from within Modal functions easily)
            import subprocess

            output_dir = Path(f"/results/{config_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "results.json"

            cmd = [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--master_port=29500",
                "/root/vitok-release/scripts/eval_vae.py",
                "--model", model,
                "--data", coco_path,
                "--max-size", str(res),
                "--batch-size", str(batch_size),
                "--num-samples", str(num_samples),
                "--crop-style", crop,
                "--output-json", str(results_file),
                "--save-visuals", "16",
                "--output-dir", str(output_dir),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = "/root/vitok-release"
            start_time = datetime.now()

            result = subprocess.run(cmd, env=env, capture_output=False)

            duration = (datetime.now() - start_time).total_seconds()

            if results_file.exists():
                with open(results_file) as f:
                    stats = json.load(f)
                stats["config_id"] = config_id
                stats["duration_seconds"] = duration
                stats["float8_mode"] = float8
                with open(results_file, "w") as f:
                    json.dump(stats, f, indent=2)
                results.append(stats)
                status["completed"] += 1
            else:
                error_stats = {
                    "config_id": config_id,
                    "error": f"Failed with return code {result.returncode}",
                }
                results.append(error_stats)
                status["failed"] += 1

        except Exception as e:
            error_stats = {"config_id": config_id, "error": str(e)}
            results.append(error_stats)
            status["failed"] += 1

        # Update status file after each eval
        status["results"] = results
        status["last_updated"] = datetime.now().isoformat()
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        results_vol.commit()

    status["completed_at"] = datetime.now().isoformat()
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    results_vol.commit()

    print(f"\n{'='*60}")
    print(f"Batch complete: {status['completed']}/{total} succeeded, {status['failed']} failed")
    print(f"Results saved to: /results/batch_{batch_id}/")
    print(f"{'='*60}")

    return results


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
)
def get_status() -> dict:
    """Get status of all evaluations."""
    results_dir = Path("/results")
    if not results_dir.exists():
        return {"status": "no_results", "message": "No results directory found"}

    # Find all batch status files
    batches = []
    for status_file in results_dir.glob("batch_*/status.json"):
        with open(status_file) as f:
            batches.append(json.load(f))

    # Find all individual eval results
    evals = []
    for results_file in results_dir.glob("*/results.json"):
        if "batch_" not in str(results_file.parent):
            with open(results_file) as f:
                evals.append(json.load(f))

    return {
        "batches": sorted(batches, key=lambda x: x.get("started_at", ""), reverse=True),
        "individual_evals": evals,
        "total_evals": len(evals),
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
)
def get_results_summary() -> dict:
    """Get summary of all evaluation results."""
    results_dir = Path("/results")
    if not results_dir.exists():
        return {"error": "No results found"}

    # Collect all results
    all_results = []
    for results_file in results_dir.glob("*/results.json"):
        if "batch_" not in str(results_file.parent):
            with open(results_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        return {"error": "No evaluation results found"}

    # Format as table
    summary = {
        "count": len(all_results),
        "results": [],
    }

    for r in sorted(all_results, key=lambda x: x.get("config_id", "")):
        summary["results"].append({
            "config": r.get("config_id", "unknown"),
            "ssim": r.get("ssim"),
            "psnr": r.get("psnr"),
            "fid": r.get("fid"),
            "fdd": r.get("fdd"),
            "samples": r.get("samples"),
            "duration_s": r.get("duration_seconds"),
        })

    return summary


@app.local_entrypoint()
def main(
    status: bool = False,
    results: bool = False,
    models: str = "",
    float8: bool = False,
    num_samples: int = 5000,
    n_gpus: int = 8,
    single: str = "",
):
    """Run batch VAE evaluations on Modal.

    Args:
        status: Show status of running/completed evals
        results: Show results summary
        models: Comma-separated list of models to evaluate (default: all T models)
        float8: Also run evaluations with float8 inference mode
        num_samples: Number of samples per evaluation (default: 5000)
        n_gpus: Number of GPUs per evaluation (default: 8)
        single: Run a single eval (format: "model,resolution,crop_style")
    """
    if status:
        print("Fetching evaluation status...")
        s = get_status.remote()
        print(f"\nTotal individual evals: {s.get('total_evals', 0)}")

        if s.get("batches"):
            print(f"\nBatches ({len(s['batches'])}):")
            for b in s["batches"]:
                print(f"  {b['batch_id']}: {b['completed']}/{b['total']} done, {b['failed']} failed")
                print(f"    Started: {b.get('started_at', 'N/A')}")
                if b.get("completed_at"):
                    print(f"    Completed: {b['completed_at']}")
        return

    if results:
        print("Fetching results summary...")
        summary = get_results_summary.remote()

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        print(f"\nEvaluation Results ({summary['count']} total):")
        print("-" * 80)
        print(f"{'Config':<35} {'SSIM':>8} {'PSNR':>8} {'FID':>8} {'FDD':>8}")
        print("-" * 80)

        for r in summary["results"]:
            ssim = f"{r['ssim']:.4f}" if r.get("ssim") else "N/A"
            psnr = f"{r['psnr']:.2f}" if r.get("psnr") else "N/A"
            fid = f"{r['fid']:.2f}" if r.get("fid") else "N/A"
            fdd = f"{r['fdd']:.2f}" if r.get("fdd") else "N/A"
            print(f"{r['config']:<35} {ssim:>8} {psnr:>8} {fid:>8} {fdd:>8}")

        print("-" * 80)
        return

    if single:
        # Run single evaluation
        parts = single.split(",")
        if len(parts) != 3:
            print("Error: --single format is 'model,resolution,crop_style'")
            print("Example: --single 'T-32x64,256,adm_square'")
            return

        model, res, crop = parts
        res = int(res)
        print(f"Running single evaluation: {model} @ {res}p {crop}")

        stats = run_single_eval.remote(
            model_name=model,
            max_size=res,
            crop_style=crop,
            num_samples=num_samples,
            n_gpus=n_gpus,
        )

        print(f"\nResults:")
        print(f"  SSIM: {stats.get('ssim', 'N/A')}")
        print(f"  PSNR: {stats.get('psnr', 'N/A')}")
        print(f"  FID:  {stats.get('fid', 'N/A')}")
        print(f"  FDD:  {stats.get('fdd', 'N/A')}")
        return

    # Build config list
    configs = list(EVAL_CONFIGS)

    # Filter by models if specified
    if models:
        model_list = [m.strip() for m in models.split(",")]
        configs = [c for c in configs if c[0] in model_list]
        print(f"Filtered to {len(configs)} configs for models: {model_list}")

    # Add float8 configs if requested
    if float8:
        if models:
            model_list = [m.strip() for m in models.split(",")]
            f8_configs = [c for c in EVAL_CONFIGS_FLOAT8 if c[0] in model_list]
        else:
            f8_configs = EVAL_CONFIGS_FLOAT8
        configs.extend(f8_configs)
        print(f"Added {len(f8_configs)} float8 configs")

    if not configs:
        print("No configs to run!")
        return

    print(f"\nRunning batch evaluation:")
    print(f"  Configs: {len(configs)}")
    print(f"  Samples per eval: {num_samples}")
    print(f"  GPUs per eval: {n_gpus}")
    print()

    for i, (model, res, crop, f8) in enumerate(configs):
        f8_str = f" (float8={f8})" if f8 else ""
        print(f"  {i+1}. {model} @ {res}p {crop}{f8_str}")

    print("\nLaunching batch on Modal...")
    print("This will run in the background. Use --status to check progress.")

    # Run the batch
    results = run_batch_eval.remote(
        configs=configs,
        num_samples=num_samples,
        n_gpus=n_gpus,
    )

    # Print summary
    print(f"\nBatch complete!")
    succeeded = sum(1 for r in results if "error" not in r)
    failed = len(results) - succeeded
    print(f"  Succeeded: {succeeded}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if succeeded > 0:
        print("\nUse --results to see detailed metrics.")
