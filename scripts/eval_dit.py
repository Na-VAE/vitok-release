#!/usr/bin/env python
"""Evaluate ViTok DiT generation quality.

This script evaluates the DiT model by generating images from class labels
and computing FID scores against real images.

Usage:
    # Generate samples for evaluation
    python examples/eval_dit.py \
        --ae_checkpoint checkpoints/ae.safetensors \
        --dit_checkpoint checkpoints/dit/final.pt \
        --output_dir eval_outputs/generated

    # Generate with specific classes
    python examples/eval_dit.py \
        --ae_checkpoint checkpoints/ae.safetensors \
        --dit_checkpoint checkpoints/dit/final.pt \
        --classes 207 360 387 974 \
        --samples_per_class 50

    # Generate with different CFG scales
    python examples/eval_dit.py \
        --ae_checkpoint checkpoints/ae.safetensors \
        --dit_checkpoint checkpoints/dit/final.pt \
        --cfg_scale 4.0 \
        --num_steps 50
"""

import argparse
from pathlib import Path
import random

import torch
from PIL import Image
import numpy as np

from vitok import AE, decode_variant, DiT, decode_dit_variant
from vitok.unipc import FlowUniPCMultistepScheduler
from vitok.naflex_io import postprocess_images
from safetensors.torch import load_file


# ImageNet class names for common classes
IMAGENET_CLASSES = {
    88: "macaw",
    207: "golden_retriever",
    279: "arctic_fox",
    360: "otter",
    387: "red_panda",
    417: "balloon",
    497: "church",
    574: "golf_ball",
    812: "space_shuttle",
    933: "cheeseburger",
    974: "geyser",
    980: "volcano",
}


def generate_batch(
    ae,
    dit,
    class_labels: torch.Tensor,
    cfg_scale: float,
    num_steps: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int = 1000,
) -> list:
    """Generate a batch of images."""
    batch_size = len(class_labels)

    # Calculate latent dimensions
    spatial_stride = getattr(ae, 'spatial_stride', 16)
    latent_size = image_size // spatial_stride
    num_tokens = latent_size * latent_size
    code_width = dit.code_width if hasattr(dit, 'code_width') else 64

    # Initialize noise
    latents = torch.randn(batch_size, num_tokens, code_width, device=device, dtype=torch.float32)

    # Null labels for CFG
    labels_null = torch.full_like(class_labels, num_classes)

    # Create scheduler
    scheduler = FlowUniPCMultistepScheduler(thresholding=False)
    scheduler.set_timesteps(num_steps)

    def autocast_ctx():
        return torch.autocast(device_type='cuda', dtype=dtype)

    # Sample
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_batch = t.expand(batch_size)

            if cfg_scale != 1.0:
                # Batched CFG
                x_in = torch.cat([latents, latents], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
                y_in = torch.cat([labels_null, class_labels], dim=0)

                with autocast_ctx():
                    out = dit({"z": x_in, "t": t_in, "context": y_in})

                uncond, cond = out.chunk(2, dim=0)
                v_pred = uncond + cfg_scale * (cond - uncond)
            else:
                with autocast_ctx():
                    v_pred = dit({"z": latents, "t": t_batch, "context": class_labels})

            latents = scheduler.step(v_pred.float(), t, latents, return_dict=False)[0]

        # Create decode dict
        y, x = torch.meshgrid(
            torch.arange(latent_size, device=device),
            torch.arange(latent_size, device=device),
            indexing='ij'
        )

        decode_dict = {
            'z': latents.to(dtype),
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


def create_grid(images: list, n_cols: int = 8) -> Image.Image:
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
    parser = argparse.ArgumentParser(description="Evaluate ViTok DiT")

    # Model
    parser.add_argument("--ae_checkpoint", type=str, required=True,
                        help="Path to AE checkpoint")
    parser.add_argument("--dit_checkpoint", type=str, required=True,
                        help="Path to DiT checkpoint")
    parser.add_argument("--ae_variant", type=str, default="Ld2-Ld22/1x16x64")
    parser.add_argument("--dit_variant", type=str, default="L/256")

    # Generation
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Class labels to generate (default: random)")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Total number of classes in the model")
    parser.add_argument("--samples_per_class", type=int, default=10,
                        help="Number of samples per class")
    parser.add_argument("--total_samples", type=int, default=None,
                        help="Total samples to generate (overrides samples_per_class)")
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Generated image size")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_outputs/dit",
                        help="Output directory for generated images")
    parser.add_argument("--save_grid", action="store_true",
                        help="Save image grids per class")
    parser.add_argument("--grid_cols", type=int, default=8,
                        help="Number of columns in grid")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading AE from {args.ae_checkpoint}...")
    ae_params = decode_variant(args.ae_variant)
    ae = AE(**ae_params)
    ae.load_state_dict(load_file(args.ae_checkpoint))
    ae.to(device=device, dtype=dtype)
    ae.eval()

    code_width = ae_params.get('channels_per_token', 64)

    print(f"Loading DiT from {args.dit_checkpoint}...")
    dit_params = decode_dit_variant(args.dit_variant)
    dit = DiT(**dit_params, code_width=code_width, text_dim=args.num_classes)
    dit.load_state_dict(load_file(args.dit_checkpoint))
    dit.to(device=device, dtype=dtype)
    dit.eval()

    # Determine classes to generate
    if args.classes:
        classes = args.classes
    else:
        # Use default interesting classes
        classes = list(IMAGENET_CLASSES.keys())

    print(f"\nGenerating images...")
    print(f"  Classes: {len(classes)}")
    print(f"  Samples per class: {args.samples_per_class}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Image size: {args.image_size}")

    total_generated = 0
    all_images = []

    for cls_idx, cls in enumerate(classes):
        cls_name = IMAGENET_CLASSES.get(cls, f"class{cls}")
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(exist_ok=True)

        cls_images = []
        samples_remaining = args.samples_per_class

        while samples_remaining > 0:
            batch_size = min(args.batch_size, samples_remaining)

            # Generate batch
            labels = torch.full((batch_size,), cls, dtype=torch.long, device=device)
            images = generate_batch(
                ae=ae,
                dit=dit,
                class_labels=labels,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                image_size=args.image_size,
                device=device,
                dtype=dtype,
                num_classes=args.num_classes,
            )

            # Save individual images
            for i, img in enumerate(images):
                idx = args.samples_per_class - samples_remaining + i
                filename = f"{cls_name}_{idx:04d}.png"
                img.save(cls_dir / filename)
                cls_images.append(img)

            samples_remaining -= batch_size
            total_generated += batch_size

        # Save grid for this class
        if args.save_grid and cls_images:
            grid = create_grid(cls_images, n_cols=args.grid_cols)
            grid.save(output_dir / f"grid_{cls_name}.png")

        all_images.extend(cls_images)
        print(f"  [{cls_idx + 1}/{len(classes)}] {cls_name}: {len(cls_images)} images")

    # Save overall grid
    if args.save_grid and all_images:
        # Sample images for overall grid
        grid_samples = all_images[:min(64, len(all_images))]
        grid = create_grid(grid_samples, n_cols=8)
        grid.save(output_dir / "grid_all.png")

    print(f"\n{'='*50}")
    print(f"Generation Complete")
    print(f"{'='*50}")
    print(f"Total images: {total_generated}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

    # Print instructions for FID computation
    print("\nTo compute FID score, use:")
    print(f"  python -m pytorch_fid {output_dir} /path/to/real/images")


if __name__ == "__main__":
    main()
