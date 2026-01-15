"""Fast upload of DIV8K dataset to Modal volume.

Downloads the full zip from HuggingFace once and extracts on Modal.

Usage:
    modal run scripts/modal/upload_div8k.py
    modal run scripts/modal/upload_div8k.py --check
"""

import modal

VOLUME_NAME = "vitok-data"

app = modal.App("vitok-upload-div8k")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install("huggingface_hub", "pillow", "tqdm")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=8192,
    cpu=2,
)
def download_div8k() -> dict:
    """Download DIV8K dataset from HuggingFace Hub."""
    import os
    import shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download, login

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    div8k_dir = Path("/data/div8k/train")

    # Check if already complete
    if div8k_dir.exists():
        n_existing = len(list(div8k_dir.glob("*.png")))
        if n_existing >= 1500:
            print(f"Already have {n_existing} images, skipping download")
            return {"status": "cached", "images": n_existing}

    print("Downloading DIV8K.zip from HuggingFace Hub...")

    # Download the zip file directly
    zip_path = hf_hub_download(
        repo_id="Iceclear/DIV8K_TrainingSet",
        filename="DIV8K.zip",
        repo_type="dataset",
        cache_dir="/tmp/hf_cache",
    )

    print(f"Downloaded to: {zip_path}")
    print("Extracting...")

    # Extract to temp location
    extract_dir = Path("/tmp/div8k_extract")
    extract_dir.mkdir(parents=True, exist_ok=True)

    os.system(f"unzip -q -o '{zip_path}' -d '{extract_dir}'")

    # Find where images are
    print("Finding images...")
    image_files = list(extract_dir.rglob("*.png"))
    if not image_files:
        image_files = list(extract_dir.rglob("*.jpg"))

    print(f"Found {len(image_files)} images")

    # Move to final location
    div8k_dir.mkdir(parents=True, exist_ok=True)

    print("Moving images to volume...")
    for i, src in enumerate(sorted(image_files)):
        dst = div8k_dir / f"{i:04d}.png"
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            # Convert to PNG if needed
            from PIL import Image
            img = Image.open(src)
            img.save(dst, "PNG")

        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(image_files)}")

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)

    vol.commit()

    n_images = len(list(div8k_dir.glob("*.png")))
    print(f"Done! {n_images} images in {div8k_dir}")

    return {"status": "downloaded", "images": n_images, "path": str(div8k_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=60)
def check_progress() -> dict:
    """Check how many images are downloaded."""
    from pathlib import Path

    div8k_dir = Path("/data/div8k/train")
    if not div8k_dir.exists():
        return {"images": 0, "expected": 1500, "complete": False}

    n_images = len(list(div8k_dir.glob("*.png")))
    return {
        "images": n_images,
        "expected": 1500,
        "complete": n_images >= 1500,
        "path": str(div8k_dir),
    }


@app.local_entrypoint()
def main(check: bool = False):
    """Download full DIV8K dataset.

    Args:
        check: Just check progress without downloading
    """
    if check:
        result = check_progress.remote()
        print(f"\nDIV8K Progress: {result['images']}/1500 images")
        if result['complete']:
            print("Download complete!")
        else:
            print(f"  {1500 - result['images']} images remaining")
        return

    print("Starting DIV8K download...")
    result = download_div8k.remote()

    print(f"\nStatus: {result['status']}")
    print(f"Images: {result['images']}")
    if 'path' in result:
        print(f"Path: {result['path']}")
