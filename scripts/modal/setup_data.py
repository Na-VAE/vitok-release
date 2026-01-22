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


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=7200)
def download_div8k(force: bool = False) -> dict:
    """Download DIV2K validation set for high-resolution evaluation.

    Note: DIV8K is no longer easily available, so we use DIV2K which has
    images up to 2K resolution - still much larger than COCO/ImageNet.
    """
    import os
    from pathlib import Path

    div8k_dir = Path("/data/div8k/val")

    # Check if already downloaded
    if not force and div8k_dir.exists():
        n_images = len(list(div8k_dir.glob("*.png"))) + len(list(div8k_dir.glob("*.jpg")))
        if n_images >= 100:
            print(f"DIV2K val already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(div8k_dir)}

    print("Downloading DIV2K validation set (high-res images up to 2K)...")
    div8k_dir.mkdir(parents=True, exist_ok=True)

    # Download DIV2K validation from official source
    import urllib.request
    import zipfile

    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    zip_path = div8k_dir.parent / "DIV2K_valid_HR.zip"

    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(div8k_dir.parent)

    # Move files from DIV2K_valid_HR to div8k/val
    extracted_dir = div8k_dir.parent / "DIV2K_valid_HR"
    if extracted_dir.exists():
        for f in extracted_dir.glob("*.png"):
            os.rename(f, div8k_dir / f.name)
        extracted_dir.rmdir()

    os.remove(zip_path)

    n_images = len(list(div8k_dir.glob("*.png"))) + len(list(div8k_dir.glob("*.jpg")))
    print(f"Downloaded {n_images} images to {div8k_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(div8k_dir)}


@app.function(
    image=image_with_datasets,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def download_div8k_full(force: bool = False) -> dict:
    """Download full DIV8K training set from HuggingFace (1500 images).

    Downloads from Iceclear/DIV8K_TrainingSet which contains high-resolution
    images up to 8K resolution for comprehensive evaluation.
    """
    import os
    from pathlib import Path

    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm import tqdm

    # Login with HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    div8k_dir = Path("/data/div8k/train")

    # Check if already downloaded
    if not force and div8k_dir.exists():
        n_images = len(list(div8k_dir.glob("*.png")))
        if n_images >= 1500:
            print(f"DIV8K full already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(div8k_dir)}

    print("Downloading DIV8K full training set from Iceclear/DIV8K_TrainingSet...")
    div8k_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode
    dataset = load_dataset("Iceclear/DIV8K_TrainingSet", split="train", streaming=True)

    # Save all images
    n_saved = 0
    for i, sample in enumerate(tqdm(dataset, desc="Downloading DIV8K")):
        img = sample["image"]
        img_path = div8k_dir / f"{i:04d}.png"
        img.save(img_path, "PNG")
        n_saved += 1

        # Commit periodically to avoid losing progress
        if (i + 1) % 100 == 0:
            vol.commit()
            print(f"  Saved {i + 1} images, committed to volume")

    n_images = len(list(div8k_dir.glob("*.png")))
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

    # DIV8K validation (DIV2K)
    div8k_val_dir = Path("/data/div8k/val")
    if div8k_val_dir.exists():
        n_images = len(list(div8k_val_dir.glob("*.png"))) + len(list(div8k_val_dir.glob("*.jpg")))
        size_gb = sum(f.stat().st_size for f in div8k_val_dir.iterdir() if f.is_file()) / (1024**3)
        results["div8k-val"] = {"images": n_images, "size_gb": round(size_gb, 2), "path": str(div8k_val_dir)}
    else:
        results["div8k-val"] = None

    # DIV8K full training set (1500 images)
    div8k_train_dir = Path("/data/div8k/train")
    if div8k_train_dir.exists():
        n_images = len(list(div8k_train_dir.glob("*.png")))
        size_gb = sum(f.stat().st_size for f in div8k_train_dir.iterdir() if f.is_file()) / (1024**3)
        results["div8k-train"] = {"images": n_images, "expected": 1500, "size_gb": round(size_gb, 2), "path": str(div8k_train_dir)}
    else:
        results["div8k-train"] = None

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
        dataset: Specific dataset to download (coco, imagenet, div8k, div8k-full). Default: coco only
        check: Check what's cached without downloading
        clear: Clear cached data
        force: Force re-download even if cached

    Examples:
        modal run scripts/modal/setup_data.py --dataset div8k-full  # Full 1500-image DIV8K
        modal run scripts/modal/setup_data.py --check               # Check what's cached
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
        elif ds == "div8k-full":
            result = download_div8k_full.remote(force=force)
        else:
            print(f"Unknown dataset: {ds}")
            print("Available: coco, imagenet, div8k, div8k-full")
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
