"""Download and cache evaluation datasets on Modal volume.

This script downloads datasets to a persistent Modal volume so they don't
need to be re-downloaded on each evaluation run.

Usage:
    # Download all datasets
    modal run scripts/modal/setup_data.py

    # Download specific dataset
    modal run scripts/modal/setup_data.py --dataset coco
    modal run scripts/modal/setup_data.py --dataset div8k

    # Check what's cached
    modal run scripts/modal/setup_data.py --check

    # Clear cached data
    modal run scripts/modal/setup_data.py --clear
"""

import modal

VOLUME_NAME = "vitok-data"

app = modal.App("vitok-setup-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install("huggingface_hub", "tqdm")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_coco(force: bool = False) -> dict:
    """Download COCO val2017 dataset."""
    import os
    from pathlib import Path

    coco_dir = Path("/data/coco/val2017")

    # Check if already downloaded
    if not force and coco_dir.exists():
        n_images = len(list(coco_dir.glob("*.jpg")))
        if n_images >= 5000:
            print(f"COCO val2017 already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(coco_dir)}

    print("Downloading COCO val2017...")
    coco_dir.parent.mkdir(parents=True, exist_ok=True)

    zip_path = Path("/data/coco/val2017.zip")
    os.system(f"wget -q --show-progress -O {zip_path} http://images.cocodataset.org/zips/val2017.zip")

    if zip_path.exists():
        os.system(f"unzip -q -o {zip_path} -d /data/coco")
        os.remove(zip_path)

    n_images = len(list(coco_dir.glob("*.jpg")))
    print(f"Downloaded {n_images} images to {coco_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(coco_dir)}


image_with_datasets = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install("huggingface_hub", "tqdm", "datasets", "pillow")
)


@app.function(
    image=image_with_datasets,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def download_imagenet1k_val(force: bool = False) -> dict:
    """Download ImageNet-1k validation set from HuggingFace.

    Requires:
        1. HuggingFace token with access to ILSVRC/imagenet-1k
        2. Accepted license at https://huggingface.co/datasets/ILSVRC/imagenet-1k

    Downloads the validation split (50K images) and saves as JPEG files.
    """
    import os
    from pathlib import Path

    from datasets import load_dataset
    from huggingface_hub import login

    # Login with HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        from huggingface_hub import whoami
        try:
            user_info = whoami()
            print(f"Logged in as: {user_info.get('name', 'unknown')}")
        except Exception as e:
            print(f"Could not get user info: {e}")
    else:
        print("Warning: HF_TOKEN not found in environment")

    imagenet_dir = Path("/data/imagenet/val")

    # Check if already downloaded
    if not force and imagenet_dir.exists():
        n_images = len(list(imagenet_dir.glob("*.JPEG"))) + len(list(imagenet_dir.glob("*.jpg")))
        if n_images >= 50000:
            print(f"ImageNet-1k val already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(imagenet_dir)}

    print("Downloading ImageNet-1k validation set from ILSVRC/imagenet-1k...")
    print("Note: Requires accepted license at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    imagenet_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use streaming to avoid downloading all splits
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            return {
                "status": "error",
                "error": f"Access denied: {e}",
                "images": 0,
                "path": str(imagenet_dir),
            }
        raise

    # Save images to folder (50K validation images)
    print(f"Streaming and saving validation images to {imagenet_dir}...")
    for i, sample in enumerate(dataset):
        img = sample["image"]
        # Save as JPEG with index-based name
        img_path = imagenet_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
        img.save(img_path, "JPEG", quality=95)

        if (i + 1) % 5000 == 0:
            print(f"  Saved {i + 1}/50000 images")

    n_images = len(list(imagenet_dir.glob("*.JPEG")))
    print(f"Downloaded {n_images} images to {imagenet_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(imagenet_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def download_div8k(force: bool = False) -> dict:
    """Download DIV8K validation set for high-resolution evaluation."""
    import os
    from pathlib import Path

    div8k_dir = Path("/data/div8k/val")

    # Check if already downloaded
    if not force and div8k_dir.exists():
        n_images = len(list(div8k_dir.glob("*.png"))) + len(list(div8k_dir.glob("*.jpg")))
        if n_images >= 100:
            print(f"DIV8K val already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(div8k_dir)}

    print("Downloading DIV8K validation set...")
    div8k_dir.mkdir(parents=True, exist_ok=True)

    # DIV8K is available on HuggingFace
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id = "eugenesiow/Div8K"

    files = list_repo_files(repo_id, repo_type="dataset")
    val_files = [f for f in files if "val" in f.lower() and (f.endswith(".png") or f.endswith(".jpg"))]

    if not val_files:
        # Try HR folder
        val_files = [f for f in files if "HR" in f and (f.endswith(".png") or f.endswith(".jpg"))][:200]

    print(f"Found {len(val_files)} validation images")

    for i, f in enumerate(val_files):
        if i % 50 == 0:
            print(f"Downloading {i}/{len(val_files)}...")
        local_path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
        target = div8k_dir / Path(f).name
        os.system(f"cp '{local_path}' '{target}'")

    n_images = len(list(div8k_dir.glob("*.png"))) + len(list(div8k_dir.glob("*.jpg")))
    print(f"Downloaded {n_images} images to {div8k_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(div8k_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=60)
def check_data() -> dict:
    """Check what datasets are cached."""
    from pathlib import Path

    results = {}

    # COCO
    coco_dir = Path("/data/coco/val2017")
    if coco_dir.exists():
        n_images = len(list(coco_dir.glob("*.jpg")))
        size_gb = sum(f.stat().st_size for f in coco_dir.glob("*.jpg")) / (1024**3)
        results["coco"] = {"images": n_images, "size_gb": round(size_gb, 2), "path": str(coco_dir)}
    else:
        results["coco"] = None

    # ImageNet-1k val
    imagenet_dir = Path("/data/imagenet/val")
    if imagenet_dir.exists():
        n_images = len(list(imagenet_dir.glob("*.JPEG"))) + len(list(imagenet_dir.glob("*.jpg")))
        if n_images > 0:
            size_gb = sum(f.stat().st_size for f in imagenet_dir.iterdir() if f.is_file()) / (1024**3)
            results["imagenet"] = {"images": n_images, "size_gb": round(size_gb, 2), "path": str(imagenet_dir)}
        else:
            results["imagenet"] = None
    else:
        results["imagenet"] = None

    # DIV8K
    div8k_dir = Path("/data/div8k/val")
    if div8k_dir.exists():
        n_images = len(list(div8k_dir.glob("*.png"))) + len(list(div8k_dir.glob("*.jpg")))
        size_gb = sum(f.stat().st_size for f in div8k_dir.iterdir() if f.is_file()) / (1024**3)
        results["div8k"] = {"images": n_images, "size_gb": round(size_gb, 2), "path": str(div8k_dir)}
    else:
        results["div8k"] = None

    return results


@app.function(image=image, volumes={"/data": vol}, timeout=60)
def clear_data(dataset: str | None = None) -> str:
    """Clear cached data."""
    import shutil
    from pathlib import Path

    if dataset:
        target = Path(f"/data/{dataset}")
        if target.exists():
            shutil.rmtree(target)
            vol.commit()
            return f"Cleared {dataset}"
        return f"{dataset} not found"
    else:
        # Clear all
        for d in Path("/data").iterdir():
            if d.is_dir():
                shutil.rmtree(d)
        vol.commit()
        return "Cleared all data"


@app.local_entrypoint()
def main(
    dataset: str | None = None,
    check: bool = False,
    clear: bool = False,
    force: bool = False,
):
    """Download and manage evaluation datasets.

    Args:
        dataset: Specific dataset to download (coco, imagenet, div8k). Default: coco only
        check: Check what's cached without downloading
        clear: Clear cached data
        force: Force re-download even if cached
    """
    if check:
        print("Checking cached datasets...")
        results = check_data.remote()
        print()
        print("=" * 50)
        print("Cached Datasets")
        print("=" * 50)
        for name, info in results.items():
            if info:
                print(f"\n{name.upper()}:")
                print(f"  Images: {info['images']}")
                print(f"  Size: {info['size_gb']} GB")
                print(f"  Path: {info['path']}")
            else:
                print(f"\n{name.upper()}: Not cached")
        return

    if clear:
        print(f"Clearing data: {dataset or 'all'}...")
        result = clear_data.remote(dataset)
        print(result)
        return

    # Download datasets
    # Default to coco only (imagenet requires auth, div8k is optional)
    datasets_to_download = [dataset] if dataset else ["coco"]

    print(f"Setting up datasets: {', '.join(datasets_to_download)}")
    print("This may take a while on first run...\n")

    for ds in datasets_to_download:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds.upper()}")
        print("=" * 50)

        if ds == "coco":
            result = download_coco.remote(force=force)
        elif ds == "imagenet":
            result = download_imagenet1k_val.remote(force=force)
        elif ds == "div8k":
            result = download_div8k.remote(force=force)
        else:
            print(f"Unknown dataset: {ds}")
            print("Available: coco, imagenet, div8k")
            continue

        print(f"Status: {result['status']}")
        if result["status"] == "error":
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"Images: {result['images']}")
            print(f"Path: {result['path']}")
        if "note" in result:
            print(f"Note: {result['note']}")

    print("\n" + "=" * 50)
    print("Dataset setup complete!")
    print("=" * 50)
    print("\nUsage in eval_vae.py:")
    print("  --dataset coco-val            # COCO val2017 (5K images)")
    print("  --dataset imagenet-val        # ImageNet-1k val (50K images, requires HF auth)")
    print("  --dataset div8k               # DIV8K (high-res)")
