#!/usr/bin/env python
"""Generate academic comparison visuals between ViTok and other VAEs.

This script creates side-by-side comparison images and computes metrics
for comparing ViTok against SD-VAE and FLUX VAE.

Usage:
    # Generate comparisons with default settings
    python scripts/generate_comparisons.py

    # Specify output directory
    python scripts/generate_comparisons.py --output assets/comparisons

    # Use specific test images
    python scripts/generate_comparisons.py --images path/to/img1.jpg path/to/img2.jpg

    # Skip certain VAEs (if you don't have them)
    python scripts/generate_comparisons.py --skip-flux

Requirements:
    pip install diffusers transformers accelerate
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ViTok imports
from vitok import AE, decode_variant, preprocess, postprocess
from vitok.pretrained import download_pretrained, get_pretrained_info


def load_vitok(model_name: str = "L-64", device: str = "cuda", dtype=torch.bfloat16):
    """Load ViTok model."""
    _, _, variant = get_pretrained_info(model_name)
    result = download_pretrained(model_name)
    weights_paths = result if isinstance(result, list) else [result]

    # Load and merge weights
    weights = {}
    for path in weights_paths:
        weights.update(load_file(path))

    model = AE(**decode_variant(variant))
    model.load_state_dict(weights)
    model.to(device=device, dtype=dtype)
    model.eval()

    spatial_stride = int(variant.split("/")[1].split("x")[1])
    return model, spatial_stride


def load_sd_vae(device: str = "cuda", dtype=torch.float16):
    """Load Stable Diffusion VAE (MIT license)."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def load_flux_vae(device: str = "cuda", dtype=torch.bfloat16):
    """Load FLUX.1 VAE (Apache 2.0 license)."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.to(device=device)
    vae.eval()
    return vae


def reconstruct_vitok(model, image: Image.Image, spatial_stride: int, device: str, dtype) -> np.ndarray:
    """Reconstruct image using ViTok."""
    pp_string = f"to_tensor|normalize(minus_one_to_one)|patchify({spatial_stride}, 256)"
    patch_dict = preprocess(image, pp=pp_string, device=device)
    patch_dict = {
        k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in patch_dict.items()
    }

    with torch.no_grad():
        encoded = model.encode(patch_dict)
        decoded = model.decode(encoded)

    images = postprocess(decoded, output_format="0_255", do_unpack=True, patch=spatial_stride)
    return images[0].permute(1, 2, 0).cpu().numpy()


def reconstruct_diffusers_vae(vae, image: Image.Image, device: str, dtype) -> np.ndarray:
    """Reconstruct image using a diffusers VAE (SD or FLUX)."""
    # Resize to multiple of 8 for VAE
    w, h = image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Convert to tensor [-1, 1]
    img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        recon = vae.decode(latent).sample

    # Convert back to numpy [0, 255]
    recon_np = recon[0].permute(1, 2, 0).cpu().float().numpy()
    recon_np = ((recon_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    return recon_np


def compute_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict:
    """Compute SSIM and PSNR metrics."""
    # Ensure same size
    if original.shape != reconstruction.shape:
        original = cv2.resize(original, (reconstruction.shape[1], reconstruction.shape[0]))

    psnr = peak_signal_noise_ratio(original, reconstruction, data_range=255)
    ssim = structural_similarity(original, reconstruction, channel_axis=2, data_range=255)

    return {"ssim": ssim, "psnr": psnr}


def compute_diff_heatmap(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Generate difference heatmap."""
    if original.shape != reconstruction.shape:
        original = cv2.resize(original, (reconstruction.shape[1], reconstruction.shape[0]))

    diff = np.abs(original.astype(np.float32) - reconstruction.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    max_diff = max(diff_gray.max(), 1.0)
    diff_normalized = (diff_gray / max_diff * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def create_comparison_grid(
    original: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    metrics: dict[str, dict],
    output_path: Path,
    title: str = "",
    cell_size: int = 256,
):
    """Create a labeled comparison grid image.

    Layout: Original | VAE1 | VAE2 | ...
    With metrics below each reconstruction.
    """
    n_cols = 1 + len(reconstructions)
    padding = 10
    label_height = 50

    # Calculate grid dimensions
    grid_width = cell_size * n_cols + padding * (n_cols - 1)
    grid_height = cell_size + label_height

    # Create grid
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    x_offset = 0

    # Add original
    orig_resized = cv2.resize(original, (cell_size, cell_size))
    grid.paste(Image.fromarray(orig_resized), (x_offset, 0))
    draw.text((x_offset + 5, cell_size + 5), "Original", fill=(0, 0, 0), font=font)
    x_offset += cell_size + padding

    # Add reconstructions
    for name, recon in reconstructions.items():
        recon_resized = cv2.resize(recon, (cell_size, cell_size))
        grid.paste(Image.fromarray(recon_resized), (x_offset, 0))

        # Add label with metrics
        m = metrics.get(name, {})
        label = f"{name}"
        metric_text = f"SSIM: {m.get('ssim', 0):.3f} | PSNR: {m.get('psnr', 0):.1f}"

        draw.text((x_offset + 5, cell_size + 3), label, fill=(0, 0, 0), font=font)
        draw.text((x_offset + 5, cell_size + 22), metric_text, fill=(80, 80, 80), font=font_small)

        x_offset += cell_size + padding

    grid.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def get_test_images() -> list[Image.Image]:
    """Get default test images from skimage."""
    from skimage import data

    images = []

    # Astronaut - good for faces and fine detail
    images.append(("astronaut", Image.fromarray(data.astronaut())))

    # Coffee - good for textures
    images.append(("coffee", Image.fromarray(data.coffee())))

    # Camera - classic test image
    camera = data.camera()
    # Convert grayscale to RGB
    camera_rgb = np.stack([camera] * 3, axis=-1)
    images.append(("camera", Image.fromarray(camera_rgb)))

    return images


def main():
    parser = argparse.ArgumentParser(description="Generate VAE comparison visuals")
    parser.add_argument("--output", type=str, default="assets/comparisons", help="Output directory")
    parser.add_argument("--images", nargs="+", help="Custom test images")
    parser.add_argument("--vitok-model", type=str, default="L-64", help="ViTok model variant")
    parser.add_argument("--skip-sd", action="store_true", help="Skip SD-VAE")
    parser.add_argument("--skip-flux", action="store_true", help="Skip FLUX VAE")
    parser.add_argument("--cell-size", type=int, default=256, help="Size of each cell in grid")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load models
    print("\nLoading ViTok...")
    vitok_model, vitok_stride = load_vitok(args.vitok_model, device, dtype)

    vaes = {"ViTok": (vitok_model, vitok_stride)}

    if not args.skip_sd:
        print("Loading SD-VAE...")
        try:
            sd_vae = load_sd_vae(device, torch.float16 if device == "cuda" else torch.float32)
            vaes["SD-VAE"] = sd_vae
        except Exception as e:
            print(f"  Warning: Could not load SD-VAE: {e}")

    if not args.skip_flux:
        print("Loading FLUX VAE...")
        try:
            flux_vae = load_flux_vae(device, dtype)
            vaes["FLUX"] = flux_vae
        except Exception as e:
            print(f"  Warning: Could not load FLUX VAE: {e}")

    # Get test images
    if args.images:
        test_images = [(Path(p).stem, Image.open(p).convert("RGB")) for p in args.images]
    else:
        print("\nUsing default test images from skimage...")
        test_images = get_test_images()

    # Process each image
    all_metrics = {}

    for img_name, img in test_images:
        print(f"\nProcessing: {img_name}")
        original_np = np.array(img)

        reconstructions = {}
        metrics = {}

        for vae_name, vae_data in vaes.items():
            print(f"  Reconstructing with {vae_name}...")

            if vae_name == "ViTok":
                model, stride = vae_data
                recon = reconstruct_vitok(model, img, stride, device, dtype)
            else:
                recon = reconstruct_diffusers_vae(vae_data, img, device, dtype)

            reconstructions[vae_name] = recon
            metrics[vae_name] = compute_metrics(original_np, recon)
            print(f"    SSIM: {metrics[vae_name]['ssim']:.4f}, PSNR: {metrics[vae_name]['psnr']:.2f}")

        # Create comparison grid
        grid_path = output_dir / f"{img_name}_comparison.png"
        create_comparison_grid(
            original_np,
            reconstructions,
            metrics,
            grid_path,
            title=img_name,
            cell_size=args.cell_size,
        )

        # Create diff heatmaps for ViTok
        if "ViTok" in reconstructions:
            heatmap = compute_diff_heatmap(original_np, reconstructions["ViTok"])
            heatmap_path = output_dir / f"{img_name}_vitok_diff.png"
            Image.fromarray(heatmap).save(heatmap_path)
            print(f"  Saved heatmap: {heatmap_path}")

        all_metrics[img_name] = metrics

    # Save metrics summary
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Image':<15} {'VAE':<10} {'SSIM':>8} {'PSNR':>8}")
    print("-" * 60)
    for img_name, img_metrics in all_metrics.items():
        for vae_name, m in img_metrics.items():
            print(f"{img_name:<15} {vae_name:<10} {m['ssim']:>8.4f} {m['psnr']:>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
