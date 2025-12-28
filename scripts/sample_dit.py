#!/usr/bin/env python
"""DiT sampling script for image generation.

Example usage:
    # Generate images with class labels
    python scripts/sample_dit.py \
        --ae_checkpoint path/to/ae.safetensors \
        --dit_checkpoint path/to/dit.safetensors \
        --classes 207 360 387 \
        --output_dir samples/

    # Generate with different CFG scales
    python scripts/sample_dit.py \
        --ae_checkpoint path/to/ae.safetensors \
        --dit_checkpoint path/to/dit.safetensors \
        --classes 207 \
        --cfg_scale 2.0 4.0 8.0 \
        --num_samples 4
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from vitok import AEConfig, load_ae
from vitok import DiTConfig, load_dit
from vitok.diffusion import FlowMatchingScheduler
from vitok.diffusion.flow_matching import euler_sample
from vitok import postprocess_images


def sample_images(
    ae,
    dit,
    class_labels: List[int],
    num_samples: int = 1,
    cfg_scale: float = 4.0,
    num_steps: int = 50,
    image_size: int = 256,
    seed: Optional[int] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Image.Image]:
    """Generate images from class labels.

    Args:
        ae: Autoencoder model
        dit: DiT model
        class_labels: List of class indices
        num_samples: Number of samples per class
        cfg_scale: Classifier-free guidance scale
        num_steps: Number of denoising steps
        image_size: Output image size
        seed: Random seed
        device: Target device
        dtype: Model dtype

    Returns:
        List of PIL images
    """
    if device is None:
        device = next(dit.parameters()).device

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Prepare labels
    labels = torch.tensor(class_labels, device=device).repeat_interleave(num_samples)
    batch_size = len(labels)

    # Calculate latent dimensions
    spatial_stride = getattr(ae, 'spatial_stride', 16)
    latent_size = image_size // spatial_stride
    num_tokens = latent_size * latent_size
    code_width = dit.code_width if hasattr(dit, 'code_width') else 64

    # Initialize noise
    z = torch.randn(batch_size, num_tokens, code_width, device=device, dtype=torch.float32)

    # Create scheduler
    scheduler = FlowMatchingScheduler()

    # Sample with flow matching
    def autocast_ctx():
        return torch.autocast(device_type='cuda', dtype=dtype)

    with torch.no_grad():
        z_denoised = euler_sample(
            dit,
            scheduler,
            z,
            labels,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            device=device,
            autocast_ctx=autocast_ctx,
        )

        # Create decode dict
        y, x = torch.meshgrid(
            torch.arange(latent_size, device=device),
            torch.arange(latent_size, device=device),
            indexing='ij'
        )

        decode_dict = {
            'z': z_denoised.to(dtype),
            'ptype': torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device),
            'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1),
            'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1),
            'original_height': torch.full((batch_size,), image_size, device=device),
            'original_width': torch.full((batch_size,), image_size, device=device),
        }

        with autocast_ctx():
            decoded = ae.decode(decode_dict)

        images = postprocess_images(
            decoded,
            output_format="0_255",
            return_type="tensor",
            unpack=True,
            spatial_stride=spatial_stride,
        )

    # Convert to PIL
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            pil_images.append(Image.fromarray(img_np))
        else:
            pil_images.append(img)

    return pil_images


def main():
    parser = argparse.ArgumentParser(description="Generate images with DiT")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--dit_checkpoint", type=str, required=True, help="Path to DiT checkpoint")
    parser.add_argument("--ae_variant", type=str, default="Ld2-Ld22/1x16x64", help="AE variant")
    parser.add_argument("--dit_variant", type=str, default="L/256", help="DiT variant")
    parser.add_argument("--classes", type=int, nargs="+", default=[207, 360, 387, 974, 88],
                        help="ImageNet class indices")
    parser.add_argument("--num_samples", type=int, default=4, help="Samples per class")
    parser.add_argument("--cfg_scale", type=float, nargs="+", default=[4.0],
                        help="CFG scale(s)")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Loading AE from {args.ae_checkpoint}...")
    ae_config = AEConfig(variant=args.ae_variant, variational=True)
    ae = load_ae(args.ae_checkpoint, ae_config, device=device, dtype=dtype)

    # Get code width from AE
    code_width = ae.encoder_width if hasattr(ae, 'encoder_width') else 64

    print(f"Loading DiT from {args.dit_checkpoint}...")
    dit_config = DiTConfig(
        variant=args.dit_variant,
        code_width=code_width,
        num_classes=args.num_classes,
    )
    dit = load_dit(args.dit_checkpoint, dit_config, device=device, dtype=dtype)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each CFG scale
    for cfg in args.cfg_scale:
        print(f"\nGenerating with CFG scale {cfg}...")

        images = sample_images(
            ae=ae,
            dit=dit,
            class_labels=args.classes,
            num_samples=args.num_samples,
            cfg_scale=cfg,
            num_steps=args.num_steps,
            image_size=args.image_size,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )

        # Save images
        for i, (img, cls_idx) in enumerate(zip(images, [c for c in args.classes for _ in range(args.num_samples)])):
            filename = f"class{cls_idx:04d}_cfg{cfg:.1f}_sample{i % args.num_samples}.png"
            img.save(output_dir / filename)
            print(f"  Saved {filename}")

        # Create grid
        if len(images) > 1:
            n_cols = min(args.num_samples, 8)
            n_rows = (len(images) + n_cols - 1) // n_cols

            grid_w = n_cols * args.image_size
            grid_h = n_rows * args.image_size
            grid = Image.new('RGB', (grid_w, grid_h))

            for i, img in enumerate(images):
                row = i // n_cols
                col = i % n_cols
                grid.paste(img, (col * args.image_size, row * args.image_size))

            grid_filename = f"grid_cfg{cfg:.1f}.png"
            grid.save(output_dir / grid_filename)
            print(f"  Saved {grid_filename}")

    print(f"\nDone! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
