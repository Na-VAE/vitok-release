#!/usr/bin/env python
"""Example: Encode and decode images with ViTok AE.

This example shows how to:
1. Load a pretrained autoencoder
2. Encode images to latent representations
3. Decode latents back to images
4. Compare original vs reconstructed images

Usage:
    python examples/encode_decode.py --checkpoint path/to/ae.safetensors --image path/to/image.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from vitok import AEConfig, load_ae
from vitok import preprocess_images, postprocess_images


def main():
    parser = argparse.ArgumentParser(description="Encode and decode images with ViTok AE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant (default: Ld2-Ld22/1x16x64)")
    parser.add_argument("--output", type=str, default="reconstructed.png",
                        help="Output image path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_size", type=int, default=512, help="Max image size")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading AE from {args.checkpoint}...")
    config = AEConfig(
        variant=args.variant,
        variational=True,
    )
    ae = load_ae(args.checkpoint, config, device=device, dtype=dtype)

    # Load and preprocess image
    print(f"Loading image from {args.image}...")
    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    print(f"  Original size: {original_size}")

    # Preprocess to NaFlex format
    spatial_stride = ae.spatial_stride if hasattr(ae, 'spatial_stride') else 16
    patch_dict = preprocess_images(
        image,
        spatial_stride=spatial_stride,
        max_size=args.max_size,
        device=str(device),
    )

    # Move to correct dtype
    patch_dict = {
        k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in patch_dict.items()
    }

    print("Encoding...")
    with torch.no_grad():
        # Encode
        encoded = ae.encode(patch_dict)

        # Get latent representation
        if 'posterior' in encoded and hasattr(encoded['posterior'], 'mode'):
            z = encoded['posterior'].mode()  # Use deterministic mode
        else:
            z = encoded['z']

        print(f"  Latent shape: {z.shape}")
        print(f"  Latent range: [{z.min():.2f}, {z.max():.2f}]")

        # Decode
        print("Decoding...")
        decode_dict = {
            'z': z,
            'ptype': patch_dict['ptype'],
            'yidx': patch_dict['yidx'],
            'xidx': patch_dict['xidx'],
            'original_height': patch_dict['original_height'],
            'original_width': patch_dict['original_width'],
        }
        decoded = ae.decode(decode_dict)

    # Post-process to images
    reconstructed = postprocess_images(
        decoded,
        output_format="0_255",
        return_type="tensor",
        unpack=True,
        spatial_stride=spatial_stride,
    )

    # Save reconstructed image
    if isinstance(reconstructed, list):
        recon_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
    else:
        recon_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)

    recon_img = Image.fromarray(recon_np.astype(np.uint8))
    recon_img.save(args.output)
    print(f"Saved reconstructed image to {args.output}")

    # Create side-by-side comparison
    comparison_path = Path(args.output).with_suffix(".comparison.png")

    # Resize original to match reconstructed
    original_resized = image.resize(recon_img.size, Image.LANCZOS)

    # Create comparison
    comparison = Image.new('RGB', (original_resized.width * 2, original_resized.height))
    comparison.paste(original_resized, (0, 0))
    comparison.paste(recon_img, (original_resized.width, 0))
    comparison.save(comparison_path)
    print(f"Saved comparison to {comparison_path}")

    # Compute reconstruction metrics
    orig_np = np.array(original_resized).astype(float)
    recon_np_float = recon_np.astype(float)

    mse = np.mean((orig_np - recon_np_float) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

    print(f"\nReconstruction metrics:")
    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
