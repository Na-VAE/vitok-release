#!/usr/bin/env python
"""Train ViTok DiT (Diffusion Transformer).

This script trains the DiT model for class-conditional image generation
using flow matching.

Usage:
    # Train on local tar shards
    python examples/train_dit.py \
        --data /path/to/shards/*.tar \
        --ae_checkpoint checkpoints/ae.safetensors \
        --output_dir checkpoints/dit

    # Train on HuggingFace dataset
    python examples/train_dit.py \
        --data hf://ILSVRC/imagenet-1k/train/*.tar \
        --ae_checkpoint checkpoints/ae.safetensors \
        --output_dir checkpoints/dit

    # Resume training
    python examples/train_dit.py \
        --data /path/to/shards \
        --ae_checkpoint checkpoints/ae.safetensors \
        --dit_checkpoint checkpoints/dit/latest.pt \
        --output_dir checkpoints/dit
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vitok import AEConfig, load_ae
from vitok import DiTConfig, create_dit, load_dit
from vitok.data import create_dataloader
from vitok.diffusion import FlowMatchingScheduler


def compute_loss(dit, ae, scheduler, batch, device, dtype):
    """Compute flow matching loss."""
    # Encode images to latents
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=torch.cuda.is_available()):
            encoded = ae.encode(batch)
            z = encoded['z']

    # Sample random timesteps
    batch_size = z.shape[0]
    t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)

    # Sample noise
    noise = torch.randn_like(z)

    # Add noise to latents
    z_noisy = scheduler.add_noise(z, noise, t)

    # Get velocity target
    v_target = scheduler.get_velocity_target(z, noise)

    # Get class labels (if available in batch)
    if 'label' in batch:
        context = batch['label'].to(device)
    else:
        # Random class labels for unconditional training
        num_classes = dit.text_dim if hasattr(dit, 'text_dim') else 1000
        context = torch.randint(0, num_classes, (batch_size,), device=device)

    # Forward pass
    with torch.autocast(device_type='cuda', dtype=dtype, enabled=torch.cuda.is_available()):
        v_pred = dit({
            'z': z_noisy,
            't': t,
            'context': context,
        })

    # Compute MSE loss on velocity
    loss = F.mse_loss(v_pred.float(), v_target.float())

    return loss


def save_checkpoint(model, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('step', 0)


def main():
    parser = argparse.ArgumentParser(description="Train ViTok DiT")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Data source (local path or hf://repo/pattern)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256, help="Image size (square crop)")
    parser.add_argument("--patch_size", type=int, default=16)

    # AE model (frozen encoder)
    parser.add_argument("--ae_checkpoint", type=str, required=True,
                        help="Path to pretrained AE checkpoint")
    parser.add_argument("--ae_variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant")

    # DiT model
    parser.add_argument("--dit_variant", type=str, default="L/256",
                        help="DiT variant (e.g., L/256, XL/512, G/256)")
    parser.add_argument("--dit_checkpoint", type=str, default=None,
                        help="Resume DiT training from checkpoint")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of class labels")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Probability of dropping class label (for CFG)")

    # Flow matching
    parser.add_argument("--shift", type=float, default=1.0,
                        help="Flow matching time shift")

    # Training
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging
    parser.add_argument("--output_dir", type=str, default="checkpoints/dit")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=10000)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen AE
    print(f"Loading AE from {args.ae_checkpoint}...")
    ae_config = AEConfig(variant=args.ae_variant, variational=True)
    ae = load_ae(args.ae_checkpoint, ae_config, device=device, dtype=dtype)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Get code width from AE
    code_width = ae.encoder_width if hasattr(ae, 'encoder_width') else 64
    print(f"AE code width: {code_width}")

    # Create DiT
    print(f"Creating DiT model: {args.dit_variant}")
    dit_config = DiTConfig(
        variant=args.dit_variant,
        code_width=code_width,
        num_classes=args.num_classes + 1,  # +1 for null class (CFG)
    )
    dit = create_dit(dit_config)
    dit.to(device=device, dtype=dtype)
    dit.train()

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling DiT...")
        dit = torch.compile(dit)

    # Count parameters
    n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print(f"DiT parameters: {n_params / 1e6:.1f}M")

    # Create dataloader (square crop for DiT)
    print(f"Loading data from: {args.data}")
    num_tokens = (args.image_size // args.patch_size) ** 2
    pp_string = (
        f"center_crop({args.image_size})|"
        f"flip|"
        f"to_tensor|"
        f"normalize(minus_one_to_one)|"
        f"patchify({args.image_size}, {args.patch_size}, {num_tokens})"
    )
    print(f"Preprocessing: {pp_string}")

    loader = create_dataloader(
        source=args.data,
        pp=pp_string,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        return_labels=True,
    )

    # Create scheduler
    scheduler = FlowMatchingScheduler(shift=args.shift)

    # Optimizer
    optimizer = AdamW(
        dit.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # Resume if checkpoint provided
    start_step = 0
    if args.dit_checkpoint:
        print(f"Resuming from {args.dit_checkpoint}")
        start_step = load_checkpoint(dit, optimizer, lr_scheduler, args.dit_checkpoint)
        print(f"Resumed at step {start_step}")

    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    loader_iter = iter(loader)
    step = start_step
    log_loss = 0
    log_count = 0
    start_time = time.time()

    while step < args.steps:
        try:
            batch, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch, labels = next(loader_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if labels is not None:
            batch['label'] = labels.to(device)

        # Convert to training dtype
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)

        # Apply CFG dropout (replace some labels with null class)
        if 'label' in batch and args.cfg_dropout > 0:
            mask = torch.rand(batch['label'].shape[0], device=device) < args.cfg_dropout
            batch['label'] = batch['label'].clone()
            batch['label'][mask] = args.num_classes  # null class

        # Forward + backward
        optimizer.zero_grad()

        loss = compute_loss(dit, ae, scheduler, batch, device, dtype)
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)

        optimizer.step()
        lr_scheduler.step()

        # Logging
        log_loss += loss.item()
        log_count += 1

        step += 1

        if step % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_loss = log_loss / log_count
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{args.steps} | "
                  f"loss: {avg_loss:.4f} | "
                  f"lr: {lr:.2e} | "
                  f"time: {elapsed:.1f}s")
            log_loss = 0
            log_count = 0
            start_time = time.time()

        if step % args.save_freq == 0:
            ckpt_path = output_dir / f"step_{step:07d}.pt"
            save_checkpoint(dit, optimizer, lr_scheduler, step, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Also save as latest
            latest_path = output_dir / "latest.pt"
            save_checkpoint(dit, optimizer, lr_scheduler, step, latest_path)

    # Final save
    final_path = output_dir / "final.pt"
    save_checkpoint(dit, optimizer, lr_scheduler, step, final_path)
    print(f"\nTraining complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
