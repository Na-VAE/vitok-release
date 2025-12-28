#!/usr/bin/env python
"""Train ViTok VAE (Autoencoder).

This script trains the ViTok autoencoder on image datasets.

Usage:
    # Train on local tar shards
    python examples/train_vae.py \
        --data /path/to/shards/*.tar \
        --output_dir checkpoints/vae

    # Train on HuggingFace dataset
    python examples/train_vae.py \
        --data hf://ILSVRC/imagenet-1k/train/*.tar \
        --output_dir checkpoints/vae

    # Resume training
    python examples/train_vae.py \
        --data /path/to/shards \
        --checkpoint checkpoints/vae/latest.pt \
        --output_dir checkpoints/vae
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vitok import AEConfig, create_ae, load_ae
from vitok.data import create_dataloader
from vitok.naflex_io import unpatchify


def compute_loss(model, batch, kl_weight=1e-6):
    """Compute reconstruction + KL loss."""
    # Forward pass
    output = model(batch, sample_posterior=True)

    # Reconstruction loss (L1)
    pred_patches = output['patches']
    target_patches = batch['patches']
    ptype = batch['ptype']

    # Mask out padding
    mask = ptype.unsqueeze(-1).float()
    recon_loss = F.l1_loss(pred_patches * mask, target_patches * mask, reduction='sum')
    recon_loss = recon_loss / mask.sum()

    # KL loss (if variational)
    kl_loss = torch.tensor(0.0, device=pred_patches.device)
    if hasattr(output.get('posterior', None), 'kl'):
        kl = output['posterior'].kl()
        kl_loss = kl.mean()

    total_loss = recon_loss + kl_weight * kl_loss

    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
    }


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
    parser = argparse.ArgumentParser(description="Train ViTok VAE")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Data source (local path or hf://repo/pattern)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=512, help="Max image size")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)

    # Model
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant (e.g., B/1x16x64, Ld2-Ld22/1x16x64)")
    parser.add_argument("--variational", action="store_true", help="Use variational AE")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    # Training
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging
    parser.add_argument("--output_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5000)

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

    # Create model
    print(f"Creating AE model: {args.variant}")
    config = AEConfig(
        variant=args.variant,
        variational=args.variational,
    )
    model = create_ae(config)
    model.to(device=device, dtype=dtype)
    model.train()

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Create dataloader
    print(f"Loading data from: {args.data}")
    pp_string = (
        f"random_resized_crop({args.max_size})|"
        f"flip|"
        f"to_tensor|"
        f"normalize(minus_one_to_one)|"
        f"patchify({args.max_size}, {args.patch_size}, {args.max_tokens})"
    )
    print(f"Preprocessing: {pp_string}")

    loader = create_dataloader(
        source=args.data,
        pp=pp_string,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # Resume if checkpoint provided
    start_step = 0
    if args.checkpoint:
        print(f"Resuming from {args.checkpoint}")
        start_step = load_checkpoint(model, optimizer, scheduler, args.checkpoint)
        print(f"Resumed at step {start_step}")

    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    loader_iter = iter(loader)
    step = start_step
    log_losses = {'loss': 0, 'recon_loss': 0, 'kl_loss': 0}
    log_count = 0
    start_time = time.time()

    while step < args.steps:
        try:
            batch, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch, _ = next(loader_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Convert to training dtype
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)

        # Forward + backward
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=dtype, enabled=torch.cuda.is_available()):
            losses = compute_loss(model, batch, kl_weight=args.kl_weight)

        loss = losses['loss']
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        # Logging
        log_losses['loss'] += losses['loss'].item()
        log_losses['recon_loss'] += losses['recon_loss'].item()
        log_losses['kl_loss'] += losses['kl_loss'].item()
        log_count += 1

        step += 1

        if step % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_losses = {k: v / log_count for k, v in log_losses.items()}
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}/{args.steps} | "
                  f"loss: {avg_losses['loss']:.4f} | "
                  f"recon: {avg_losses['recon_loss']:.4f} | "
                  f"kl: {avg_losses['kl_loss']:.6f} | "
                  f"lr: {lr:.2e} | "
                  f"time: {elapsed:.1f}s")
            log_losses = {'loss': 0, 'recon_loss': 0, 'kl_loss': 0}
            log_count = 0
            start_time = time.time()

        if step % args.save_freq == 0:
            ckpt_path = output_dir / f"step_{step:07d}.pt"
            save_checkpoint(model, optimizer, scheduler, step, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Also save as latest
            latest_path = output_dir / "latest.pt"
            save_checkpoint(model, optimizer, scheduler, step, latest_path)

    # Final save
    final_path = output_dir / "final.pt"
    save_checkpoint(model, optimizer, scheduler, step, final_path)
    print(f"\nTraining complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
