#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

This script evaluates the VAE reconstruction quality on a dataset,
computing metrics like PSNR, SSIM, and LPIPS.

Usage:
    # Evaluate on local images
    python examples/eval_vae.py \
        --data /path/to/images/*.jpg \
        --checkpoint checkpoints/vae/final.pt

    # Evaluate on HuggingFace dataset
    python examples/eval_vae.py \
        --data hf://ILSVRC/imagenet-1k/val/*.tar \
        --checkpoint checkpoints/vae/final.pt \
        --save_samples eval_outputs/

    # Evaluate with different configurations
    python examples/eval_vae.py \
        --data /path/to/images \
        --checkpoint checkpoints/vae.safetensors \
        --max_size 512 \
        --max_tokens 1024
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from vitok import AEConfig, load_ae
from vitok import preprocess_images, postprocess_images


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]

    Returns:
        Average PSNR in dB
    """
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    return psnr.mean().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute Structural Similarity Index.

    Simplified SSIM implementation.

    Args:
        pred: Predicted images [B, C, H, W] in range [0, 1]
        target: Target images [B, C, H, W] in range [0, 1]

    Returns:
        Average SSIM
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Simple local means using average pooling
    kernel_size = window_size
    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred ** 2, kernel_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target ** 2, kernel_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, kernel_size, stride=1, padding=padding) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

    return ssim_map.mean().item()


def load_images(path: str, max_images: Optional[int] = None) -> list:
    """Load images from a directory or glob pattern."""
    path = Path(path)

    if path.is_file():
        return [Image.open(path).convert('RGB')]

    if path.is_dir():
        files = sorted(path.glob('*'))
    else:
        # Glob pattern
        parent = path.parent
        pattern = path.name
        files = sorted(parent.glob(pattern))

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_files = [f for f in files if f.suffix.lower() in image_extensions]

    if max_images:
        image_files = image_files[:max_images]

    images = []
    for f in image_files:
        try:
            img = Image.open(f).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    return images


def save_comparison(original: Image.Image, reconstructed: Image.Image, path: Path):
    """Save side-by-side comparison image."""
    # Resize to same size for comparison
    w1, h1 = original.size
    w2, h2 = reconstructed.size

    # Use the larger dimensions
    w, h = max(w1, w2), max(h1, h2)

    canvas = Image.new('RGB', (w * 2 + 10, h), (128, 128, 128))
    canvas.paste(original, (0, 0))
    canvas.paste(reconstructed, (w + 10, 0))

    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to images or glob pattern")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to evaluate")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint")
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant")
    parser.add_argument("--variational", action="store_true",
                        help="Use variational mode (sample from posterior)")

    # Preprocessing
    parser.add_argument("--max_size", type=int, default=512,
                        help="Maximum image size")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens")

    # Output
    parser.add_argument("--save_samples", type=str, default=None,
                        help="Directory to save sample reconstructions")
    parser.add_argument("--num_save", type=int, default=10,
                        help="Number of samples to save")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    print(f"Loading AE from {args.checkpoint}...")
    config = AEConfig(
        variant=args.variant,
        variational=args.variational,
    )
    ae = load_ae(args.checkpoint, config, device=device, dtype=dtype)
    ae.eval()

    # Get spatial stride
    spatial_stride = ae.spatial_stride if hasattr(ae, 'spatial_stride') else args.patch_size

    # Load images
    print(f"Loading images from {args.data}...")
    images = load_images(args.data, args.max_images)
    print(f"Loaded {len(images)} images")

    if len(images) == 0:
        print("No images found!")
        return

    # Create save directory if needed
    if args.save_samples:
        save_dir = Path(args.save_samples)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Build preprocessing string
    pp_string = (
        f"to_tensor|"
        f"normalize(minus_one_to_one)|"
        f"patchify({args.max_size}, {args.patch_size}, {args.max_tokens})"
    )

    # Metrics accumulators
    total_psnr = 0.0
    total_ssim = 0.0
    total_l1 = 0.0
    num_images = 0
    saved_count = 0

    print(f"\nEvaluating...")
    print(f"Preprocessing: {pp_string}")

    # Process in batches
    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i + args.batch_size]

        # Preprocess
        batch = preprocess_images(batch_images, pp=pp_string, device=device)

        # Convert to training dtype
        batch['patches'] = batch['patches'].to(dtype)

        # Encode and decode
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=torch.cuda.is_available()):
                output = ae(batch, sample_posterior=args.variational)

        # Postprocess reconstructions
        recon_images = postprocess_images(
            output,
            output_format="zero_to_one",
            current_format="minus_one_to_one",
            unpack=True,
            patch=spatial_stride,
        )

        # Convert original images to tensors for comparison
        for j, (orig_pil, recon_tensor) in enumerate(zip(batch_images, recon_images)):
            # Resize original to match reconstruction size
            _, h, w = recon_tensor.shape
            orig_resized = orig_pil.resize((w, h), Image.LANCZOS)
            orig_tensor = torch.from_numpy(np.array(orig_resized)).permute(2, 0, 1).float() / 255.0
            orig_tensor = orig_tensor.to(device)

            # Ensure same shape
            recon_tensor = recon_tensor.to(device)

            # Compute metrics
            psnr = compute_psnr(recon_tensor.unsqueeze(0), orig_tensor.unsqueeze(0))
            ssim = compute_ssim(recon_tensor.unsqueeze(0), orig_tensor.unsqueeze(0))
            l1 = F.l1_loss(recon_tensor, orig_tensor).item()

            total_psnr += psnr
            total_ssim += ssim
            total_l1 += l1
            num_images += 1

            # Save sample comparisons
            if args.save_samples and saved_count < args.num_save:
                recon_np = (recon_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                recon_pil = Image.fromarray(recon_np)

                save_path = save_dir / f"comparison_{saved_count:03d}.png"
                save_comparison(orig_resized, recon_pil, save_path)
                print(f"  Saved: {save_path}")
                saved_count += 1

        # Progress
        if (i + args.batch_size) % (args.batch_size * 10) == 0 or i + args.batch_size >= len(images):
            print(f"  Processed {min(i + args.batch_size, len(images))}/{len(images)} images")

    # Compute averages
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_l1 = total_l1 / num_images

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_images} images)")
    print(f"{'='*50}")
    print(f"PSNR:  {avg_psnr:.2f} dB")
    print(f"SSIM:  {avg_ssim:.4f}")
    print(f"L1:    {avg_l1:.4f}")
    print(f"{'='*50}")

    if args.save_samples:
        print(f"\nSample reconstructions saved to: {save_dir}")


if __name__ == "__main__":
    main()
