#!/usr/bin/env python
"""Generate visual comparisons for blog.

Creates side-by-side reconstructions and close-up crops comparing
ViTok models against baseline VAEs.

Usage:
    # Generate all comparison categories
    modal run scripts/eval/visuals.py

    # Specific category
    modal run scripts/eval/visuals.py --category text

    # Local run
    python scripts/eval/visuals.py --data /path/to/images --output-dir results/visuals
"""
import argparse
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


CATEGORIES = ["text", "details", "foliage", "architecture"]


def load_image(path: Path, max_size: int | None = 512) -> torch.Tensor:
    """Load and preprocess image.

    Args:
        path: Path to image file
        max_size: If set, resize longest side to this. If None, use native resolution.

    Returns tensor in [0, 1] range with shape [C, H, W].
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max_size is not None:
        # Resize longest side
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        # Native resolution
        new_w, new_h = w, h

    # Round to multiple of 8 for VAE compatibility
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return TF.to_tensor(img)


def reconstruct_vitok(image: torch.Tensor, model_name: str, device: str = "cuda") -> torch.Tensor:
    """Reconstruct image using ViTok model."""
    from vitok import AE, decode_variant
    from vitok.pretrained import load_pretrained
    from vitok.pp.ops import patchify, unpatchify

    pretrained = load_pretrained(model_name)
    config = decode_variant(pretrained["variant"])

    encoder = AE(**config, decoder=False).to(device, dtype=torch.float16).eval()
    encoder.load_state_dict(pretrained["encoder"], strict=False)

    decoder = AE(**config, encoder=False).to(device, dtype=torch.float16).eval()
    decoder.load_state_dict(pretrained["decoder"], strict=False)

    patch_size = encoder.spatial_stride
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
    out_patches = decoded["patches"]  # [1, N, P*P*3]
    out_patches = out_patches.reshape(1, grid_h, grid_w, 3, patch_size, patch_size)
    out_patches = out_patches.permute(0, 3, 1, 4, 2, 5).reshape(1, 3, h, w)

    # Convert back to [0, 1]
    recon = (out_patches / 2 + 0.5).clamp(0, 1)
    return recon[0].cpu().float()


def reconstruct_baseline(image: torch.Tensor, model_name: str, device: str = "cuda") -> torch.Tensor:
    """Reconstruct image using baseline VAE."""
    from scripts.eval.baselines import BaselineVAE

    vae = BaselineVAE(model_name, device=device, dtype=torch.float16)

    img_batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        recon = vae.encode_decode(img_batch)

    return recon[0].cpu().float()


def compute_diff(original: torch.Tensor, reconstructed: torch.Tensor, amplify: float = 5.0) -> torch.Tensor:
    """Compute amplified difference image."""
    diff = (original - reconstructed).abs() * amplify
    return diff.clamp(0, 1)


def create_comparison_grid(
    original: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    output_path: Path,
    include_diff: bool = True,
):
    """Create side-by-side comparison grid.

    Layout: Original | Model1 | Model2 | ... [| Diff]
    """
    images = [TF.to_pil_image(original)]
    labels = ["Original"]

    for name, recon in reconstructions.items():
        images.append(TF.to_pil_image(recon))
        labels.append(name)

    if include_diff and len(reconstructions) > 0:
        # Add diff for first reconstruction
        first_recon = list(reconstructions.values())[0]
        diff = compute_diff(original, first_recon)
        images.append(TF.to_pil_image(diff))
        labels.append("Diff (5x)")

    # Create grid
    w, h = images[0].size
    n = len(images)
    grid = Image.new("RGB", (w * n, h))

    for i, img in enumerate(images):
        grid.paste(img, (i * w, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)
    return labels


def create_closeup(
    original: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    crop_box: tuple[int, int, int, int],  # (x, y, w, h)
    output_path: Path,
):
    """Create close-up crop comparison.

    Args:
        crop_box: (x, y, width, height) in pixels
    """
    x, y, cw, ch = crop_box

    # Crop original
    orig_crop = original[:, y:y+ch, x:x+cw]
    images = [TF.to_pil_image(orig_crop)]

    # Crop reconstructions
    for name, recon in reconstructions.items():
        crop = recon[:, y:y+ch, x:x+cw]
        images.append(TF.to_pil_image(crop))

    # Create grid
    w, h = images[0].size
    n = len(images)
    grid = Image.new("RGB", (w * n, h))

    for i, img in enumerate(images):
        grid.paste(img, (i * w, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)


def generate_visuals(
    image_paths: list[Path],
    output_dir: Path,
    vitok_models: list[str] = None,
    baseline_models: list[str] = None,
    max_size: int | None = 512,
    device: str = "cuda",
):
    """Generate visual comparisons for a set of images.

    Args:
        image_paths: List of image paths to process
        output_dir: Directory to save outputs
        vitok_models: List of ViTok model names
        baseline_models: List of baseline VAE names (flux, sd, qwen)
        max_size: Max image size, or None for native resolution
        device: Device to run on
    """
    if vitok_models is None:
        vitok_models = ["350M-f16x64"]
    if baseline_models is None:
        baseline_models = ["flux", "sd", "qwen"]

    res_str = "native" if max_size is None else f"{max_size}px"
    print(f"Processing {len(image_paths)} images at {res_str} resolution")

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_dir / "comparison_grids"
    closeup_dir = output_dir / "closeups"

    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load image
        original = load_image(img_path, max_size=max_size)
        _, h, w = original.shape

        # Generate reconstructions
        reconstructions = {}

        for model in vitok_models:
            try:
                recon = reconstruct_vitok(original, model, device=device)
                reconstructions[f"ViTok-{model}"] = recon
            except Exception as e:
                print(f"Error with ViTok {model}: {e}")

        for model in baseline_models:
            try:
                recon = reconstruct_baseline(original, model, device=device)
                reconstructions[model.upper()] = recon
            except Exception as e:
                print(f"Error with baseline {model}: {e}")

        # Save comparison grid
        stem = img_path.stem
        create_comparison_grid(
            original,
            reconstructions,
            comparison_dir / f"{stem}_comparison.jpg",
        )

        # Create center close-up (200x200)
        crop_size = 200
        cx, cy = w // 2, h // 2
        crop_box = (cx - crop_size // 2, cy - crop_size // 2, crop_size, crop_size)
        create_closeup(
            original,
            reconstructions,
            crop_box,
            closeup_dir / f"{stem}_closeup_center.jpg",
        )

        # Create corner close-up
        crop_box = (0, 0, crop_size, crop_size)
        create_closeup(
            original,
            reconstructions,
            crop_box,
            closeup_dir / f"{stem}_closeup_corner.jpg",
        )

        # Clear GPU memory between images
        torch.cuda.empty_cache()

    print(f"\nVisuals saved to: {output_dir}")


def main():
    from scripts.modal.modal_config import DATASET_PATHS

    parser = argparse.ArgumentParser(description="Generate visual comparisons")
    parser.add_argument("--data", type=Path, help="Path to images directory (local run)")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_PATHS.keys()),
                        help="Benchmark dataset to use (Modal run)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/visuals"))
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process (default: all)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Max image size (default: native resolution)")
    parser.add_argument("--native-res", action="store_true",
                        help="Use native resolution (same as --max-size unset)")
    parser.add_argument("--vitok-models", nargs="+", default=["5B-f16x64"],
                        help="ViTok models to compare")
    parser.add_argument("--baseline-models", nargs="+", default=["flux", "sd", "qwen"],
                        help="Baseline VAEs to compare")
    parser.add_argument("--modal", action="store_true", help="Run on Modal")
    args = parser.parse_args()

    # Native resolution by default
    max_size = None if args.native_res or args.max_size is None else args.max_size

    if args.modal:
        import modal
        from scripts.modal.modal_config import image, gpu, data_vol, output_vol, hf_secret

        app = modal.App("vitok-visuals")

        @app.function(
            image=image,
            gpu="H100",
            volumes={"/data": data_vol, "/output": output_vol},
            secrets=[hf_secret],
            timeout=7200,  # 2 hours for large datasets at native res
        )
        def _run(dataset: str, max_images: int | None, max_size: int | None,
                 vitok_models: list[str], baseline_models: list[str]):
            import sys
            import os
            sys.path.insert(0, "/root/vitok-release")
            os.environ["HF_HOME"] = "/data/huggingface"

            # Get dataset path
            data_path = Path(DATASET_PATHS.get(dataset, f"/data/benchmarks/{dataset}"))
            if not data_path.exists():
                raise ValueError(f"Dataset not found: {data_path}")

            # Find images
            image_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                image_paths.extend(data_path.glob(ext))
            image_paths = sorted(image_paths)

            if max_images:
                image_paths = image_paths[:max_images]

            print(f"Dataset: {dataset} ({len(image_paths)} images)")
            print(f"Resolution: {'native' if max_size is None else f'{max_size}px'}")
            print(f"ViTok models: {vitok_models}")
            print(f"Baseline models: {baseline_models}")

            output_dir = Path(f"/output/visuals/benchmarks/{dataset}")

            generate_visuals(
                image_paths=image_paths,
                output_dir=output_dir,
                vitok_models=vitok_models,
                baseline_models=baseline_models,
                max_size=max_size,
            )

            output_vol.commit()
            print(f"\nSaved to: {output_dir}")

        # Default to challenge dataset
        dataset = args.dataset or "challenge"

        with app.run():
            _run.remote(
                dataset=dataset,
                max_images=args.max_images,
                max_size=max_size,
                vitok_models=args.vitok_models,
                baseline_models=args.baseline_models,
            )

    else:
        if not args.data:
            parser.error("--data is required for local run")

        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_paths.extend(args.data.glob(ext))
        image_paths = sorted(image_paths)

        if args.max_images:
            image_paths = image_paths[:args.max_images]

        generate_visuals(
            image_paths=image_paths,
            output_dir=args.output_dir,
            vitok_models=args.vitok_models,
            baseline_models=args.baseline_models,
            max_size=max_size,
        )


if __name__ == "__main__":
    main()
