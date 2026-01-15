#!/usr/bin/env python
"""Generate visual assets for ViTok-v2 blog post.

Creates:
1. Original + reconstruction images for all models
2. L1 error heatmaps
3. Category-organized samples
4. Metadata JSON for interactive HTML

Usage:
    # Generate all blog visuals on Modal
    modal run scripts/eval/generate_blog_visuals.py

    # Generate specific category
    modal run scripts/eval/generate_blog_visuals.py --category foliage

    # Local run
    python scripts/eval/generate_blog_visuals.py --data /path/to/images --output-dir results/blog
"""
import argparse
import json
import colorsys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Image categories with HuggingFace dataset sources for streaming
CATEGORIES = {
    "foliage": {
        "description": "Nature/landscape - tests fine texture reconstruction",
        "hf_dataset": "eugenesiow/Div2k",
        "hf_config": "bicubic_x2",
        "hf_split": "validation",
        "hf_image_key": "hr",
    },
    "portraits": {
        "description": "Portraits - tests perceptual quality on faces",
        "hf_dataset": "nielsr/CelebA-faces",
        "hf_config": None,
        "hf_split": "train",
        "hf_image_key": "image",
    },
    "architecture": {
        "description": "Buildings - tests geometric edge preservation",
        "hf_dataset": "huggan/wikiart",
        "hf_config": None,
        "hf_split": "train",
        "hf_image_key": "image",
    },
    "animals": {
        "description": "Animals - tests fur/texture detail",
        "hf_dataset": "Bingsu/Cat_and_Dog",
        "hf_config": None,
        "hf_split": "train",
        "hf_image_key": "image",
    },
    "general": {
        "description": "General diverse images from COCO",
        "hf_dataset": "detection-datasets/coco",
        "hf_config": None,
        "hf_split": "val",
        "hf_image_key": "image",
    },
}

# All ViTok models to compare
ALL_MODELS = [
    "350M-f16x16", "350M-f16x32", "350M-f16x64",
    "5B-f16x16", "5B-f16x32", "5B-f16x64",
    "5B-f32x64", "5B-f32x128",
]

# Baseline VAEs for comparison
BASELINE_MODELS = ["flux", "sd", "qwen"]


def load_image(path: Path, max_size: int = 512) -> torch.Tensor:
    """Load and preprocess image to [C, H, W] in [0, 1] range."""
    img = Image.open(path).convert("RGB")

    # Resize longest side, preserving aspect ratio
    w, h = img.size
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Round to multiple of 16 for VAE compatibility (largest patch size)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    img = img.resize((new_w, new_h), Image.LANCZOS)

    return TF.to_tensor(img)


def reconstruct_vitok(image: torch.Tensor, model_name: str, device: str = "cuda") -> torch.Tensor:
    """Reconstruct image using ViTok model."""
    from vitok import AE, decode_variant
    from vitok.pretrained import load_pretrained

    pretrained = load_pretrained(model_name)
    config = decode_variant(pretrained["variant"])
    patch_size = config.get("patch", 16)

    encoder = AE(**config, decoder=False).to(device, dtype=torch.float16).eval()
    encoder.load_state_dict(pretrained["encoder"], strict=False)

    decoder = AE(**config, encoder=False).to(device, dtype=torch.float16).eval()
    decoder.load_state_dict(pretrained["decoder"], strict=False)

    _, h, w = image.shape
    grid_h, grid_w = h // patch_size, w // patch_size

    # Convert to [-1, 1] and patchify
    img_norm = image * 2 - 1
    img_batch = img_norm.unsqueeze(0).to(device, dtype=torch.float16)

    # Create patch dict
    patches = img_batch.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, patch_size * patch_size * 3)

    row_idx = torch.arange(grid_h, device=device).repeat_interleave(grid_w).unsqueeze(0)
    col_idx = torch.arange(grid_w, device=device).repeat(grid_h).unsqueeze(0)

    batch = {
        "patches": patches,
        "patch_sizes": torch.tensor([[grid_h, grid_w]], device=device),
        "row_idx": row_idx,
        "col_idx": col_idx,
    }

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        encoded = encoder.encode(batch)
        decoded = decoder.decode(encoded)

    # Unpatchify
    out_patches = decoded["patches"]
    out_patches = out_patches.reshape(1, grid_h, grid_w, 3, patch_size, patch_size)
    out_patches = out_patches.permute(0, 3, 1, 4, 2, 5).reshape(1, 3, h, w)

    # Convert back to [0, 1]
    recon = (out_patches / 2 + 0.5).clamp(0, 1)
    return recon[0].cpu().float()


def reconstruct_baseline(image: torch.Tensor, model_name: str, device: str = "cuda") -> torch.Tensor:
    """Reconstruct image using baseline VAE (Flux, SD, Qwen)."""
    from scripts.eval.baselines import BaselineVAE

    vae = BaselineVAE(model_name, device=device, dtype=torch.float16)
    recon = vae.encode_decode(image.unsqueeze(0))
    return recon[0].cpu().float()


def compute_l1_heatmap(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute L1 error heatmap as RGB tensor.

    Returns a viridis-style heatmap where:
    - Blue = low error
    - Yellow = high error
    """
    # Compute per-pixel L1 error (average across channels)
    l1_error = (original - reconstructed).abs().mean(dim=0)  # [H, W]

    # Normalize to [0, 1] using 99th percentile for better contrast
    vmax = torch.quantile(l1_error, 0.99).item()
    vmax = max(vmax, 0.01)  # Avoid division by zero
    l1_normalized = (l1_error / vmax).clamp(0, 1)

    # Convert to viridis-style colormap
    h, w = l1_normalized.shape
    heatmap = torch.zeros(3, h, w)

    # Simple viridis-like mapping (blue -> green -> yellow)
    for y in range(h):
        for x in range(w):
            v = l1_normalized[y, x].item()
            # Viridis approximation
            if v < 0.5:
                r = 0.267 + v * 0.6
                g = 0.004 + v * 0.8
                b = 0.329 + v * 0.3
            else:
                r = 0.267 + 0.3 + (v - 0.5) * 1.0
                g = 0.004 + 0.4 + (v - 0.5) * 0.6
                b = 0.329 + 0.15 - (v - 0.5) * 0.5

            heatmap[0, y, x] = min(1.0, r)
            heatmap[1, y, x] = min(1.0, g)
            heatmap[2, y, x] = max(0.0, b)

    return heatmap


def compute_l1_heatmap_fast(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute L1 error heatmap using vectorized operations."""
    import numpy as np

    # Compute per-pixel L1 error (average across channels)
    l1_error = (original - reconstructed).abs().mean(dim=0).numpy()  # [H, W]

    # Normalize to [0, 1] using 99th percentile
    vmax = np.percentile(l1_error, 99)
    vmax = max(vmax, 0.01)
    l1_normalized = np.clip(l1_error / vmax, 0, 1)

    # Apply matplotlib-style viridis colormap
    h, w = l1_normalized.shape
    heatmap = np.zeros((3, h, w), dtype=np.float32)

    # Viridis colormap approximation (vectorized)
    v = l1_normalized
    heatmap[0] = np.clip(0.267 + v * 0.9, 0, 1)  # R
    heatmap[1] = np.clip(0.05 + v * 0.85, 0, 1)  # G
    heatmap[2] = np.clip(0.33 - v * 0.3, 0, 1)   # B

    return torch.from_numpy(heatmap)


def save_image(tensor: torch.Tensor, path: Path, quality: int = 95):
    """Save tensor as image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = TF.to_pil_image(tensor.clamp(0, 1))
    img.save(path, quality=quality)


def stream_images_from_hf(category: str, max_images: int = 20, max_size: int = 512):
    """Stream images from HuggingFace dataset for a category.

    Yields (image_id, tensor) tuples where tensor is [C, H, W] in [0, 1] range.
    """
    from datasets import load_dataset

    cat_info = CATEGORIES.get(category)
    if not cat_info:
        raise ValueError(f"Unknown category: {category}")

    dataset_name = cat_info["hf_dataset"]
    config = cat_info.get("hf_config")
    split = cat_info["hf_split"]
    image_key = cat_info["hf_image_key"]

    print(f"Streaming from {dataset_name} ({split})...")

    # Load with streaming
    if config:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    count = 0
    for i, sample in enumerate(dataset):
        if count >= max_images:
            break

        try:
            img = sample[image_key]
            if img is None:
                continue

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize longest side
            w, h = img.size
            if max(w, h) < 256:  # Skip tiny images
                continue

            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            # Round to multiple of 32 (for all patch sizes)
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            if new_w < 32 or new_h < 32:
                continue

            img = img.resize((new_w, new_h), Image.LANCZOS)
            tensor = TF.to_tensor(img)

            yield f"{category}_{count:04d}", tensor
            count += 1

        except Exception as e:
            print(f"  Skipping sample {i}: {e}")
            continue

    print(f"  Streamed {count} images for {category}")


def generate_blog_assets(
    image_paths: list[Path],
    output_dir: Path,
    models: list[str] = None,
    baselines: list[str] = None,
    max_size: int = 512,
    device: str = "cuda",
    category: str = "misc",
):
    """Generate all blog assets for a set of images."""
    if models is None:
        models = ALL_MODELS
    if baselines is None:
        baselines = BASELINE_MODELS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    originals_dir = output_dir / "originals"
    recons_dir = output_dir / "reconstructions"
    heatmaps_dir = output_dir / "heatmaps"

    all_models = models + baselines  # Combine ViTok and baseline models

    metadata = {
        "category": category,
        "description": CATEGORIES.get(category, {}).get("description", ""),
        "images": [],
        "vitok_models": models,
        "baseline_models": baselines,
        "max_size": max_size,
    }

    for img_path in tqdm(image_paths, desc=f"Processing {category}"):
        stem = img_path.stem

        # Load original
        original = load_image(img_path, max_size=max_size)
        _, h, w = original.shape

        # Save original
        orig_path = originals_dir / f"{stem}.jpg"
        save_image(original, orig_path)

        image_meta = {
            "id": stem,
            "original": str(orig_path.relative_to(output_dir)),
            "width": w,
            "height": h,
            "reconstructions": {},
            "heatmaps": {},
        }

        # Generate reconstructions for ViTok models
        for model in models:
            model_safe = model.replace("-", "_")
            try:
                recon = reconstruct_vitok(original, model, device=device)
                _save_recon_and_heatmap(
                    original, recon, model, model_safe, stem,
                    recons_dir, heatmaps_dir, output_dir, image_meta
                )
            except Exception as e:
                print(f"Error with ViTok {model} on {stem}: {e}")
            torch.cuda.empty_cache()

        # Generate reconstructions for baseline models
        for baseline in baselines:
            try:
                recon = reconstruct_baseline(original, baseline, device=device)
                _save_recon_and_heatmap(
                    original, recon, baseline, baseline, stem,
                    recons_dir, heatmaps_dir, output_dir, image_meta
                )
            except Exception as e:
                print(f"Error with baseline {baseline} on {stem}: {e}")
            torch.cuda.empty_cache()

        metadata["images"].append(image_meta)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(metadata['images'])} images to {output_dir}")
    return metadata


def _save_recon_and_heatmap(original, recon, model_name, model_safe, stem,
                            recons_dir, heatmaps_dir, output_dir, image_meta):
    """Helper to save reconstruction and heatmap for a model."""
    # Save reconstruction
    recon_path = recons_dir / model_safe / f"{stem}.jpg"
    save_image(recon, recon_path)
    image_meta["reconstructions"][model_name] = str(recon_path.relative_to(output_dir))

    # Compute and save L1 heatmap
    heatmap = compute_l1_heatmap_fast(original, recon)
    heatmap_path = heatmaps_dir / model_safe / f"{stem}.jpg"
    save_image(heatmap, heatmap_path)
    image_meta["heatmaps"][model_name] = str(heatmap_path.relative_to(output_dir))

    # Compute metrics for this image
    psnr = 10 * torch.log10(1.0 / ((original - recon) ** 2).mean()).item()
    l1 = (original - recon).abs().mean().item()
    image_meta.setdefault("metrics", {})[model_name] = {
        "psnr": round(psnr, 2),
        "l1": round(l1, 4),
    }


# Modal setup (only when running via `modal run`)
try:
    import modal
    from scripts.modal.modal_config import eval_image, with_vitok_code, hf_secret

    app = modal.App("vitok-blog-visuals")
    image = with_vitok_code(eval_image)

    weights_vol = modal.Volume.from_name("vitok-weights", create_if_missing=True)
    data_vol = modal.Volume.from_name("vitok-data", create_if_missing=True)
    results_vol = modal.Volume.from_name("vitok-eval-results", create_if_missing=True)

    @app.function(
        image=image,
        gpu="H100",
        volumes={"/cache": weights_vol, "/data": data_vol, "/results": results_vol},
        secrets=[hf_secret],
        timeout=7200,
    )
    def run_blog_visuals(
        category: str = "all",
        models: list[str] = None,
        max_size: int = 512,
        max_images: int = 5,
    ):
        """Generate blog visuals on Modal."""
        import sys
        import os
        sys.path.insert(0, "/root/vitok-release")
        os.environ["HF_HOME"] = "/cache/huggingface"

        if models is None:
            models = ALL_MODELS

        # Find images in DIV8K
        data_path = Path("/data/div8k/val")
        all_images = sorted(data_path.glob("*.png"))
        print(f"Found {len(all_images)} images in {data_path}")

        if category == "all":
            categories_to_run = list(CATEGORIES.keys())
        else:
            categories_to_run = [category]

        all_metadata = {}

        for cat in categories_to_run:
            # Use different image subsets for different categories
            cat_idx = list(CATEGORIES.keys()).index(cat)
            start_idx = cat_idx * max_images
            cat_images = all_images[start_idx:start_idx + max_images]

            if not cat_images:
                cat_images = all_images[:max_images]

            output_dir = Path("/results/blog/assets") / cat
            metadata = generate_blog_assets(
                image_paths=cat_images,
                output_dir=output_dir,
                models=models,
                max_size=max_size,
                category=cat,
            )
            all_metadata[cat] = metadata

        # Save combined metadata
        combined_path = Path("/results/blog/assets/metadata_all.json")
        with open(combined_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        results_vol.commit()
        return all_metadata

    @app.local_entrypoint()
    def main(
        category: str = "all",
        max_size: int = 512,
        max_images: int = 5,
    ):
        """Local entrypoint for Modal."""
        result = run_blog_visuals.remote(
            category=category,
            models=ALL_MODELS,
            max_size=max_size,
            max_images=max_images,
        )
        print(f"\nGenerated assets for categories: {list(result.keys())}")

except ImportError:
    app = None  # Modal not installed


def local_main():
    """Local execution entrypoint."""
    parser = argparse.ArgumentParser(description="Generate blog visual assets")
    parser.add_argument("--data", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/blog/assets"))
    parser.add_argument("--category", choices=list(CATEGORIES.keys()) + ["all"], default="all")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=5, help="Max images per category")
    args = parser.parse_args()

    image_paths = sorted(args.data.glob("*.jpg")) + sorted(args.data.glob("*.png"))
    image_paths = image_paths[:args.max_images]

    generate_blog_assets(
        image_paths=image_paths,
        output_dir=args.output_dir,
        models=args.models,
        max_size=args.max_size,
        category=args.category if args.category != "all" else "misc",
    )


if __name__ == "__main__":
    local_main()
