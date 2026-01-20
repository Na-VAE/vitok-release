"""Download high-resolution (4K+) images for evaluation.

Downloads from Unsplash which provides free high-resolution images.

Usage:
    modal run scripts/modal/download_4k_images.py --count 100
"""

import modal

app = modal.App("download-4k")
vol = modal.Volume.from_name("vitok-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl")
    .pip_install("pillow", "requests", "tqdm")
)


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def download_4k_images(count: int = 100, min_size: int = 4096) -> dict:
    """Download high-resolution images from Pexels (free API, better quality).

    Args:
        count: Number of images to download
        min_size: Minimum dimension (width or height) in pixels
    """
    import os
    import requests
    from pathlib import Path
    from PIL import Image
    import io
    import time
    import random

    output_dir = Path(f"/data/hires_{min_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    if len(existing) >= count:
        print(f"Already have {len(existing)} images in {output_dir}")

        # Verify sizes
        sizes = []
        for img_path in existing[:10]:
            with Image.open(img_path) as img:
                sizes.append(max(img.size))

        return {
            "status": "cached",
            "count": len(existing),
            "path": str(output_dir),
            "sample_sizes": sizes
        }

    print(f"Downloading {count} images with min dimension {min_size}px...")
    print(f"Using Pexels API (curated high-res photos)...")

    # Pexels API - free, provides original resolution images
    # Sign up at https://www.pexels.com/api/ for a key
    # For now, use their curated endpoint which doesn't require auth for moderate use

    downloaded = len(existing)
    page = 1

    # Search queries that tend to have high-res images
    queries = [
        "landscape 4k", "nature 8k", "architecture", "cityscape",
        "mountain", "ocean", "aerial view", "drone photography",
        "wallpaper 4k", "high resolution nature"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for query in queries:
        if downloaded >= count:
            break

        print(f"\nSearching: {query}")

        for page in range(1, 10):
            if downloaded >= count:
                break

            try:
                # Use Pexels search (works without API key for limited use)
                # Actually, let's use picsum.photos which provides high-res images
                # Or better - use Lorem Picsum with specific large sizes

                # Download from picsum (lorem picsum) - reliable, high quality
                for _ in range(5):
                    if downloaded >= count:
                        break

                    # Request specific large size
                    img_id = random.randint(1, 1000)
                    # picsum caps at certain sizes, let's try direct URLs

                    # Alternative: Use wallhaven API (has 4K/8K wallpapers)
                    # For now, download from picsum at max available size
                    url = f"https://picsum.photos/id/{img_id}/5000/5000"

                    response = requests.get(url, timeout=60, allow_redirects=True, headers=headers)

                    if response.status_code != 200:
                        continue

                    # Check image size
                    img_data = io.BytesIO(response.content)
                    try:
                        with Image.open(img_data) as img:
                            w, h = img.size
                            max_dim = max(w, h)

                            if max_dim < min_size:
                                print(f"  Skip id={img_id}: {w}x{h} (need {min_size}+)")
                                continue

                            # Save
                            filename = f"picsum_{img_id}_{w}x{h}.jpg"
                            filepath = output_dir / filename

                            if filepath.exists():
                                continue

                            # Convert to RGB if needed and save
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(filepath, "JPEG", quality=95)

                            downloaded += 1
                            print(f"[{downloaded}/{count}] Saved {filename}")
                    except Exception as e:
                        print(f"  Error processing: {e}")

                    time.sleep(0.3)

            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(1)

    # If we still don't have enough, try wallhaven (has actual 4K/8K)
    if downloaded < count:
        print("\nTrying wallhaven for 4K+ images...")

        for page in range(1, 20):
            if downloaded >= count:
                break

            try:
                # Wallhaven API - search for large resolution wallpapers
                url = f"https://wallhaven.cc/api/v1/search?categories=100&purity=100&atleast=3840x2160&sorting=random&page={page}"

                response = requests.get(url, timeout=30, headers=headers)
                if response.status_code != 200:
                    continue

                data = response.json()
                for item in data.get("data", []):
                    if downloaded >= count:
                        break

                    img_url = item.get("path")
                    resolution = item.get("resolution", "")

                    if not img_url:
                        continue

                    # Parse resolution
                    try:
                        res_w, res_h = map(int, resolution.split("x"))
                        if max(res_w, res_h) < min_size:
                            continue
                    except:
                        continue

                    # Download
                    try:
                        img_response = requests.get(img_url, timeout=60, headers=headers)
                        if img_response.status_code != 200:
                            continue

                        img_data = io.BytesIO(img_response.content)
                        with Image.open(img_data) as img:
                            w, h = img.size
                            filename = f"wallhaven_{item['id']}_{w}x{h}.jpg"
                            filepath = output_dir / filename

                            if filepath.exists():
                                continue

                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(filepath, "JPEG", quality=95)

                            downloaded += 1
                            print(f"[{downloaded}/{count}] Saved {filename} (wallhaven)")

                    except Exception as e:
                        print(f"  Error downloading {img_url}: {e}")

                    time.sleep(0.5)

            except Exception as e:
                print(f"  Wallhaven error: {e}")
                time.sleep(1)

    # Commit to volume
    vol.commit()

    # Get final stats
    final_images = list(output_dir.glob("*.jpg"))
    sizes = []
    for img_path in final_images[:20]:
        with Image.open(img_path) as img:
            sizes.append(max(img.size))

    return {
        "status": "downloaded",
        "count": len(final_images),
        "path": str(output_dir),
        "sample_sizes": sizes,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0
    }


@app.function(image=image, volumes={"/data": vol}, timeout=300)
def check_hires_images() -> dict:
    """Check existing high-res images."""
    from pathlib import Path
    from PIL import Image

    results = {}
    for d in Path("/data").iterdir():
        if d.is_dir() and "hires" in d.name:
            images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if images:
                sizes = []
                for img_path in images[:20]:
                    try:
                        with Image.open(img_path) as img:
                            sizes.append(max(img.size))
                    except:
                        pass
                results[d.name] = {
                    "count": len(images),
                    "min": min(sizes) if sizes else 0,
                    "max": max(sizes) if sizes else 0,
                    "path": str(d)
                }
    return results


@app.local_entrypoint()
def main(
    count: int = 100,
    min_size: int = 4096,
    check: bool = False,
):
    """Download high-resolution images for evaluation.

    Examples:
        # Download 100 4K+ images
        modal run scripts/modal/download_4k_images.py --count 100

        # Download 50 8K+ images
        modal run scripts/modal/download_4k_images.py --count 50 --min-size 8192

        # Check existing
        modal run scripts/modal/download_4k_images.py --check
    """
    if check:
        result = check_hires_images.remote()
        if not result:
            print("No high-res datasets found")
        else:
            for name, info in result.items():
                print(f"{name}: {info['count']} images, {info['min']}-{info['max']}px")
        return

    print(f"Downloading {count} images with min dimension {min_size}px...")
    result = download_4k_images.remote(count=count, min_size=min_size)

    print(f"\nResult: {result['status']}")
    print(f"Images: {result['count']}")
    print(f"Path: {result['path']}")
    print(f"Size range: {result.get('min_size', 'N/A')}-{result.get('max_size', 'N/A')}px")
