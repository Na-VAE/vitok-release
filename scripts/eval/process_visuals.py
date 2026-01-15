#!/usr/bin/env python
"""Process visuals.pt files from eval_vae.py into JPGs + heatmaps.

Converts Modal output to static blog assets:
    results/blog/{category}/
        originals/          # Shared original images
        recons/{model}/     # Reconstruction per model
        heatmaps/{model}/   # L1 error heatmaps
        metadata.json       # Image list + per-image metrics

Usage:
    # After downloading from Modal
    modal volume get vitok-output /blog results/blog

    # Process all categories
    python scripts/eval/process_visuals.py

    # Process specific category
    python scripts/eval/process_visuals.py --category foliage
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


CATEGORIES = ["foliage", "faces", "urban", "text", "animals", "coco"]
VITOK_MODELS = ["350M-f16x16", "350M-f16x32", "350M-f16x64", "5B-f16x16", "5B-f16x32", "5B-f16x64"]
BASELINES = ["flux", "sd", "qwen"]


def compute_l1_heatmap(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute L1 error heatmap as RGB tensor using viridis colormap."""
    # Compute per-pixel L1 error (average across channels)
    l1_error = (original - reconstructed).abs().mean(dim=0).numpy()  # [H, W]

    # Normalize to [0, 1] using 99th percentile for better contrast
    vmax = np.percentile(l1_error, 99)
    vmax = max(vmax, 0.01)
    l1_normalized = np.clip(l1_error / vmax, 0, 1)

    # Apply viridis-style colormap (simplified version)
    h, w = l1_normalized.shape
    heatmap = np.zeros((3, h, w), dtype=np.float32)

    t = l1_normalized
    # Viridis approximation: dark purple -> blue -> green -> yellow
    heatmap[0] = 0.267 + 0.003 * t + t * t * (0.993 - 0.267 - 0.003)  # R
    heatmap[1] = 0.004 + 0.873 * t - 0.364 * t * t                     # G
    heatmap[2] = 0.329 + 0.678 * t - 1.556 * t * t + 0.549 * t * t * t # B

    return torch.from_numpy(np.clip(heatmap, 0, 1))


def process_category(category: str, blog_dir: Path, verbose: bool = True):
    """Process all model visuals for a category."""
    cat_dir = blog_dir / category

    # Output directories
    originals_dir = cat_dir / "originals"
    recons_dir = cat_dir / "recons"
    heatmaps_dir = cat_dir / "heatmaps"

    originals_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "category": category,
        "images": [],
        "models": [],
    }

    # Track which images we've saved (use first model's refs as originals)
    originals_saved = False
    num_images = 0

    # Process all models
    all_models = VITOK_MODELS + [f"baseline-{b}" for b in BASELINES]

    for model_name in all_models:
        # Handle baseline naming
        if model_name.startswith("baseline-"):
            visuals_path = cat_dir / model_name.replace("baseline-", "") / "visuals.pt"
            model_label = model_name
        else:
            visuals_path = cat_dir / model_name / "visuals.pt"
            model_label = model_name

        if not visuals_path.exists():
            if verbose:
                print(f"  Skipping {model_label} (no visuals.pt)")
            continue

        if verbose:
            print(f"  Processing {model_label}...")

        data = torch.load(visuals_path, weights_only=False)
        refs = data["ref"]
        recons = data["recon"]

        # Create model output dirs
        model_safe = model_label.replace("/", "-")
        (recons_dir / model_safe).mkdir(parents=True, exist_ok=True)
        (heatmaps_dir / model_safe).mkdir(parents=True, exist_ok=True)

        metadata["models"].append(model_label)

        for i, (ref, recon) in enumerate(zip(refs, recons)):
            img_name = f"{i:04d}.jpg"

            # Save originals once (from first model)
            if not originals_saved:
                save_image(ref, originals_dir / img_name)
                metadata["images"].append({
                    "id": i,
                    "original": f"originals/{img_name}",
                    "recons": {},
                    "heatmaps": {},
                    "metrics": {},
                })

            # Save reconstruction
            save_image(recon, recons_dir / model_safe / img_name)
            metadata["images"][i]["recons"][model_label] = f"recons/{model_safe}/{img_name}"

            # Compute and save heatmap
            heatmap = compute_l1_heatmap(ref, recon)
            save_image(heatmap, heatmaps_dir / model_safe / img_name)
            metadata["images"][i]["heatmaps"][model_label] = f"heatmaps/{model_safe}/{img_name}"

            # Compute per-image metrics
            psnr = 10 * torch.log10(1.0 / ((ref - recon) ** 2).mean()).item()
            l1 = (ref - recon).abs().mean().item()
            ssim = compute_ssim(ref, recon)

            metadata["images"][i]["metrics"][model_label] = {
                "psnr": round(psnr, 2),
                "l1": round(l1, 4),
                "ssim": round(ssim, 4),
            }

        originals_saved = True
        num_images = len(refs)

    # Save metadata
    with open(cat_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  Saved {num_images} images, {len(metadata['models'])} models")

    return metadata


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM between two images (simple implementation)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Convert to grayscale for simplicity
    if img1.dim() == 3:
        img1 = img1.mean(dim=0)
        img2 = img2.mean(dim=0)

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.item()


def main():
    parser = argparse.ArgumentParser(description="Process visuals.pt to JPGs + heatmaps")
    parser.add_argument("--blog-dir", default="results/blog", help="Blog output directory")
    parser.add_argument("--category", default=None, help="Process single category (default: all)")
    args = parser.parse_args()

    blog_dir = Path(args.blog_dir)

    if args.category:
        categories = [args.category]
    else:
        categories = CATEGORIES

    print("Processing blog visuals...")
    print("=" * 50)

    for cat in categories:
        print(f"\nCategory: {cat}")
        if not (blog_dir / cat).exists():
            print(f"  Skipping (directory not found)")
            continue
        process_category(cat, blog_dir)

    print("\n" + "=" * 50)
    print("Done! Open results/blog/index.html to view")


if __name__ == "__main__":
    main()
