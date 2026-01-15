"""Download and upload standard benchmark datasets for visual comparison.

Downloads well-established image reconstruction benchmarks:
- Kodak: 24 images, 768x512, classic compression benchmark
- Set14: 14 images, various, textures and patterns
- Urban100: 100 images, various, architecture and grid structures
- BSD100: 100 images, various, natural scenes and foliage
- CelebA: 50 images, 178x218, face reconstruction

Usage:
    # Upload all datasets
    modal run scripts/modal/upload_benchmarks.py

    # Upload specific dataset
    modal run scripts/modal/upload_benchmarks.py --dataset kodak

    # Check what's cached
    modal run scripts/modal/upload_benchmarks.py --check

    # Clear benchmark data
    modal run scripts/modal/upload_benchmarks.py --clear
"""

import modal

VOLUME_NAME = "vitok-data"

app = modal.App("vitok-upload-benchmarks")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install("huggingface_hub", "tqdm", "requests", "pillow", "datasets")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Dataset URLs
DATASET_URLS = {
    "kodak": "https://r0k.us/graphics/kodak/",  # Individual PNG files
    "set14": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip",
    "urban100": "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip",
    "bsd100": "https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip",
    "celeba": "nielsr/CelebA-faces",  # HuggingFace dataset
}

# Hand-picked challenge images for website visuals
CHALLENGE_IMAGES = {
    "cosplayers": "https://i.redd.it/rw7g1xmvx1wc1.jpeg",  # Faces, costumes, details
    "chinatown": "https://www.takewalks.com/blog/wp-content/uploads/2023/06/walks_sf_day_01.jpg",  # Urban, text/signs
    "flowers": "https://i.redd.it/ktzyxj3prjz81.jpg",  # Painful colors, fine details
    "buildings": "https://static01.nyt.com/images/2018/10/05/us/05sfbuildings/merlin_136702875_6d0ee402-e952-4402-b5b3-d2a2d8668147-superJumbo.jpg",  # Architecture
    "magazines": "https://i.redd.it/zm4px3hvggj51.jpg",  # Brutal text reconstruction
}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_kodak(force: bool = False) -> dict:
    """Download Kodak Lossless True Color Image Suite (24 images)."""
    import os
    from pathlib import Path

    import requests

    kodak_dir = Path("/data/benchmarks/kodak")

    # Check if already downloaded
    if not force and kodak_dir.exists():
        n_images = len(list(kodak_dir.glob("*.png")))
        if n_images >= 24:
            print(f"Kodak already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(kodak_dir)}

    print("Downloading Kodak dataset (24 images)...")
    kodak_dir.mkdir(parents=True, exist_ok=True)

    # Download individual PNG files
    base_url = "https://r0k.us/graphics/kodak/kodak/"
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        url = f"{base_url}{filename}"
        filepath = kodak_dir / filename

        print(f"  Downloading {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            filepath.write_bytes(response.content)
        else:
            print(f"  Warning: Failed to download {filename}")

    n_images = len(list(kodak_dir.glob("*.png")))
    print(f"Downloaded {n_images} images to {kodak_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(kodak_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_set14(force: bool = False) -> dict:
    """Download Set14 benchmark dataset."""
    import os
    import shutil
    import zipfile
    from pathlib import Path

    import requests

    set14_dir = Path("/data/benchmarks/set14")

    # Check if already downloaded
    if not force and set14_dir.exists():
        n_images = len(list(set14_dir.glob("*.png"))) + len(list(set14_dir.glob("*.bmp")))
        if n_images >= 14:
            print(f"Set14 already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(set14_dir)}

    print("Downloading Set14 dataset...")
    set14_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path("/data/benchmarks/set14.zip")
    url = DATASET_URLS["set14"]

    print(f"  Downloading from {url}...")
    response = requests.get(url, allow_redirects=True)
    zip_path.write_bytes(response.content)

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(set14_dir)

    # Flatten: move all image files from subdirectories to set14_dir
    for subdir in list(set14_dir.iterdir()):
        if subdir.is_dir():
            for f in subdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in [".png", ".bmp", ".jpg", ".jpeg"]:
                    dest = set14_dir / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
            shutil.rmtree(subdir)

    if zip_path.exists():
        os.remove(zip_path)

    n_images = len(list(set14_dir.glob("*.png"))) + len(list(set14_dir.glob("*.bmp")))
    print(f"Downloaded {n_images} images to {set14_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(set14_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_urban100(force: bool = False) -> dict:
    """Download Urban100 benchmark dataset."""
    import os
    import shutil
    import zipfile
    from pathlib import Path

    import requests

    urban_dir = Path("/data/benchmarks/urban100")

    # Check if already downloaded
    if not force and urban_dir.exists():
        n_images = len(list(urban_dir.glob("*.png")))
        if n_images >= 100:
            print(f"Urban100 already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(urban_dir)}

    print("Downloading Urban100 dataset...")
    urban_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path("/data/benchmarks/urban100.zip")
    url = DATASET_URLS["urban100"]

    print(f"  Downloading from {url}...")
    response = requests.get(url, allow_redirects=True)
    zip_path.write_bytes(response.content)

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(urban_dir)

    # Flatten: move all image files from subdirectories to urban_dir
    for subdir in list(urban_dir.iterdir()):
        if subdir.is_dir():
            for f in subdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    dest = urban_dir / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
            shutil.rmtree(subdir)

    if zip_path.exists():
        os.remove(zip_path)

    n_images = len(list(urban_dir.glob("*.png")))
    print(f"Downloaded {n_images} images to {urban_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(urban_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_bsd100(force: bool = False) -> dict:
    """Download BSD100 (Berkeley Segmentation Dataset) benchmark."""
    import os
    import shutil
    import zipfile
    from pathlib import Path

    import requests

    bsd_dir = Path("/data/benchmarks/bsd100")

    # Check if already downloaded
    if not force and bsd_dir.exists():
        n_images = len(list(bsd_dir.glob("*.png"))) + len(list(bsd_dir.glob("*.jpg")))
        if n_images >= 100:
            print(f"BSD100 already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(bsd_dir)}

    print("Downloading BSD100 dataset...")
    bsd_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path("/data/benchmarks/bsd100.zip")
    url = DATASET_URLS["bsd100"]

    print(f"  Downloading from {url}...")
    response = requests.get(url, allow_redirects=True)
    zip_path.write_bytes(response.content)

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(bsd_dir)

    # Flatten: move all image files from subdirectories to bsd_dir
    for subdir in list(bsd_dir.iterdir()):
        if subdir.is_dir():
            for f in subdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    dest = bsd_dir / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
            shutil.rmtree(subdir)

    if zip_path.exists():
        os.remove(zip_path)

    n_images = len(list(bsd_dir.glob("*.png"))) + len(list(bsd_dir.glob("*.jpg")))
    print(f"Downloaded {n_images} images to {bsd_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(bsd_dir)}


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_celeba(force: bool = False, n_images: int = 50) -> dict:
    """Download CelebA faces subset from HuggingFace."""
    import os
    from pathlib import Path

    from datasets import load_dataset
    from huggingface_hub import login

    celeba_dir = Path("/data/benchmarks/celeba")

    # Check if already downloaded
    if not force and celeba_dir.exists():
        existing = len(list(celeba_dir.glob("*.png"))) + len(list(celeba_dir.glob("*.jpg")))
        if existing >= n_images:
            print(f"CelebA already cached: {existing} images")
            return {"status": "cached", "images": existing, "path": str(celeba_dir)}

    # Login with HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Downloading CelebA faces ({n_images} images)...")
    celeba_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode
    dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)

    # Save first n_images
    for i, sample in enumerate(dataset):
        if i >= n_images:
            break

        img = sample["image"]
        img_path = celeba_dir / f"celeba_{i:04d}.png"
        img.save(img_path, "PNG")

        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{n_images} images")

    saved = len(list(celeba_dir.glob("*.png")))
    print(f"Downloaded {saved} images to {celeba_dir}")

    vol.commit()
    return {"status": "downloaded", "images": saved, "path": str(celeba_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download_challenge(force: bool = False) -> dict:
    """Download hand-picked challenge images for website visuals."""
    from pathlib import Path

    import requests

    challenge_dir = Path("/data/benchmarks/challenge")

    # Check if already downloaded
    if not force and challenge_dir.exists():
        n_images = len(list(challenge_dir.glob("*.jpg"))) + len(list(challenge_dir.glob("*.jpeg"))) + len(list(challenge_dir.glob("*.png")))
        if n_images >= len(CHALLENGE_IMAGES):
            print(f"Challenge images already cached: {n_images} images")
            return {"status": "cached", "images": n_images, "path": str(challenge_dir)}

    print(f"Downloading challenge images ({len(CHALLENGE_IMAGES)} images)...")
    challenge_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for name, url in CHALLENGE_IMAGES.items():
        print(f"  Downloading {name}...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                # Determine extension from URL or content-type
                ext = ".jpg"
                if url.endswith(".png"):
                    ext = ".png"
                elif url.endswith(".jpeg"):
                    ext = ".jpeg"

                filepath = challenge_dir / f"{name}{ext}"
                filepath.write_bytes(response.content)
                print(f"    Saved: {filepath.name}")
            else:
                print(f"    Warning: Failed to download {name} (status {response.status_code})")
        except Exception as e:
            print(f"    Error downloading {name}: {e}")

    n_images = len(list(challenge_dir.glob("*.jpg"))) + len(list(challenge_dir.glob("*.jpeg"))) + len(list(challenge_dir.glob("*.png")))
    print(f"Downloaded {n_images} images to {challenge_dir}")

    vol.commit()
    return {"status": "downloaded", "images": n_images, "path": str(challenge_dir)}


@app.function(image=image, volumes={"/data": vol}, timeout=60)
def check_benchmarks() -> dict:
    """Check what benchmark datasets are cached."""
    from pathlib import Path

    results = {}
    benchmarks_dir = Path("/data/benchmarks")

    datasets = ["kodak", "set14", "urban100", "bsd100", "celeba", "challenge"]
    expected = {"kodak": 24, "set14": 14, "urban100": 100, "bsd100": 100, "celeba": 50, "challenge": 5}

    for ds in datasets:
        ds_dir = benchmarks_dir / ds
        if ds_dir.exists():
            n_images = (
                len(list(ds_dir.glob("*.png")))
                + len(list(ds_dir.glob("*.jpg")))
                + len(list(ds_dir.glob("*.bmp")))
            )
            size_mb = sum(f.stat().st_size for f in ds_dir.iterdir() if f.is_file()) / (1024**2)
            results[ds] = {
                "images": n_images,
                "expected": expected[ds],
                "complete": n_images >= expected[ds],
                "size_mb": round(size_mb, 1),
                "path": str(ds_dir),
            }
        else:
            results[ds] = None

    return results


@app.function(image=image, volumes={"/data": vol}, timeout=60)
def clear_benchmarks(dataset: str | None = None) -> str:
    """Clear cached benchmark data."""
    import shutil
    from pathlib import Path

    benchmarks_dir = Path("/data/benchmarks")

    if dataset:
        target = benchmarks_dir / dataset
        if target.exists():
            shutil.rmtree(target)
            vol.commit()
            return f"Cleared {dataset}"
        return f"{dataset} not found"
    else:
        if benchmarks_dir.exists():
            shutil.rmtree(benchmarks_dir)
            vol.commit()
            return "Cleared all benchmark data"
        return "No benchmark data to clear"


@app.local_entrypoint()
def main(
    dataset: str | None = None,
    check: bool = False,
    clear: bool = False,
    force: bool = False,
):
    """Download and manage benchmark datasets for visual comparison.

    Args:
        dataset: Specific dataset to download (kodak, set14, urban100, bsd100, celeba)
        check: Check what's cached without downloading
        clear: Clear cached data
        force: Force re-download even if cached
    """
    if check:
        print("Checking cached benchmark datasets...")
        results = check_benchmarks.remote()
        print()
        print("=" * 60)
        print("Benchmark Datasets")
        print("=" * 60)

        total_images = 0
        total_expected = 0

        for name, info in results.items():
            if info:
                status = "OK" if info["complete"] else "INCOMPLETE"
                print(f"\n{name.upper()} [{status}]:")
                print(f"  Images: {info['images']}/{info['expected']}")
                print(f"  Size: {info['size_mb']} MB")
                print(f"  Path: {info['path']}")
                total_images += info["images"]
                total_expected += info["expected"]
            else:
                print(f"\n{name.upper()}: Not cached")
                total_expected += {"kodak": 24, "set14": 14, "urban100": 100, "bsd100": 100, "celeba": 50}[name]

        print()
        print("=" * 60)
        print(f"Total: {total_images}/{total_expected} images")
        return

    if clear:
        print(f"Clearing benchmark data: {dataset or 'all'}...")
        result = clear_benchmarks.remote(dataset)
        print(result)
        return

    # Download datasets
    datasets_to_download = [dataset] if dataset else ["kodak", "set14", "urban100", "bsd100", "celeba", "challenge"]

    print(f"Uploading benchmark datasets: {', '.join(datasets_to_download)}")
    print("This may take a while on first run...\n")

    download_funcs = {
        "kodak": download_kodak,
        "set14": download_set14,
        "urban100": download_urban100,
        "bsd100": download_bsd100,
        "celeba": download_celeba,
        "challenge": download_challenge,
    }

    for ds in datasets_to_download:
        print(f"\n{'=' * 50}")
        print(f"Dataset: {ds.upper()}")
        print("=" * 50)

        if ds not in download_funcs:
            print(f"Unknown dataset: {ds}")
            print("Available: kodak, set14, urban100, bsd100, celeba")
            continue

        result = download_funcs[ds].remote(force=force)
        print(f"Status: {result['status']}")
        print(f"Images: {result['images']}")
        print(f"Path: {result['path']}")

    print("\n" + "=" * 50)
    print("Benchmark upload complete!")
    print("=" * 50)
    print("\nUsage:")
    print("  modal run scripts/modal/upload_benchmarks.py --check")
