#!/usr/bin/env python
"""Example: Generate images with DiT.

This example shows how to:
1. Load pretrained AE and DiT models
2. Generate images from class labels
3. Apply classifier-free guidance
4. Create image grids

Usage:
    python examples/dit_generation.py \
        --ae_checkpoint path/to/ae.safetensors \
        --dit_checkpoint path/to/dit.safetensors \
        --classes 207 360 387 \
        --cfg_scale 4.0

ImageNet class examples:
    207: golden retriever
    360: otter
    387: lesser panda (red panda)
    974: geyser
    88: macaw
    279: arctic fox
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from vitok import AEConfig, load_ae
from vitok import DiTConfig, load_dit
from vitok.diffusion import FlowMatchingScheduler
from vitok.diffusion.flow_matching import euler_sample
from vitok import postprocess_images


# ImageNet class names for common classes
IMAGENET_CLASSES = {
    88: "macaw",
    207: "golden retriever",
    279: "arctic fox",
    360: "otter",
    387: "red panda",
    417: "balloon",
    497: "church",
    574: "golf ball",
    812: "space shuttle",
    933: "cheeseburger",
    974: "geyser",
    980: "volcano",
}


def generate_images(
    ae,
    dit,
    class_labels: list,
    num_samples: int = 4,
    cfg_scale: float = 4.0,
    num_steps: int = 50,
    image_size: int = 256,
    seed: int = 42,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Generate images from class labels."""
    if device is None:
        device = next(dit.parameters()).device

    torch.manual_seed(seed)

    # Prepare labels
    labels = torch.tensor(class_labels, device=device).repeat_interleave(num_samples)
    batch_size = len(labels)

    # Calculate latent dimensions
    spatial_stride = getattr(ae, 'spatial_stride', 16)
    latent_size = image_size // spatial_stride
    num_tokens = latent_size * latent_size
    code_width = dit.code_width if hasattr(dit, 'code_width') else 64

    print(f"Generating {batch_size} images...")
    print(f"  Latent grid: {latent_size}x{latent_size} = {num_tokens} tokens")
    print(f"  Code width: {code_width}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Steps: {num_steps}")

    # Initialize noise
    z = torch.randn(batch_size, num_tokens, code_width, device=device, dtype=torch.float32)

    # Create scheduler
    scheduler = FlowMatchingScheduler()

    def autocast_ctx():
        return torch.autocast(device_type='cuda', dtype=dtype)

    # Sample
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

        # Decode
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
            img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        else:
            pil_images.append(img)

    return pil_images


def create_grid(images: list, n_cols: int = 4) -> Image.Image:
    """Create image grid from list of PIL images."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    w, h = images[0].size
    grid = Image.new('RGB', (n_cols * w, n_rows * h), (255, 255, 255))

    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        grid.paste(img, (col * w, row * h))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate images with DiT")
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--dit_checkpoint", type=str, required=True)
    parser.add_argument("--ae_variant", type=str, default="Ld2-Ld22/1x16x64")
    parser.add_argument("--dit_variant", type=str, default="L/256")
    parser.add_argument("--classes", type=int, nargs="+", default=[207, 360, 387, 974])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load models
    print(f"Loading AE from {args.ae_checkpoint}...")
    ae_config = AEConfig(variant=args.ae_variant, variational=True)
    ae = load_ae(args.ae_checkpoint, ae_config, device=device, dtype=dtype)

    code_width = ae.encoder_width if hasattr(ae, 'encoder_width') else 64

    print(f"Loading DiT from {args.dit_checkpoint}...")
    dit_config = DiTConfig(
        variant=args.dit_variant,
        code_width=code_width,
        num_classes=args.num_classes,
    )
    dit = load_dit(args.dit_checkpoint, dit_config, device=device, dtype=dtype)

    # Generate
    images = generate_images(
        ae=ae,
        dit=dit,
        class_labels=args.classes,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        image_size=args.image_size,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving images to {output_dir}...")

    # Save individual images
    for i, (img, cls) in enumerate(zip(images, [c for c in args.classes for _ in range(args.num_samples)])):
        cls_name = IMAGENET_CLASSES.get(cls, f"class{cls}")
        filename = f"{cls_name}_{i % args.num_samples}.png"
        img.save(output_dir / filename)
        print(f"  {filename}")

    # Create and save grid
    grid = create_grid(images, n_cols=args.num_samples)
    grid_path = output_dir / "grid.png"
    grid.save(grid_path)
    print(f"  grid.png")

    print(f"\nDone! Generated {len(images)} images.")


if __name__ == "__main__":
    main()
