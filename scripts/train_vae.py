#!/usr/bin/env python
"""Train ViTok VAE (Autoencoder).

Supports FSDP2/DDP distributed training with perceptual losses.

Usage:
    # Local single GPU
    python scripts/train_vae.py --data /path/to/shards/*.tar --output_dir checkpoints/vae

    # Local multi-GPU with FSDP2
    torchrun --nproc_per_node=8 scripts/train_vae.py \
        --data hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar --fsdp

    # Modal cloud GPU (8x A100, recommended)
    modal run scripts/train_vae.py --steps 100000 --wandb-project vitok

    # Modal with custom data
    modal run scripts/train_vae.py \
        --data hf://ILSVRC/imagenet-1k/train-{00000..01023}.tar \
        --variant Ld2-Ld22/1x16x64 --steps 50000

    # Modal finetune from pretrained
    modal run scripts/train_vae.py --pretrained 350M-f16x64 --freeze-encoder --steps 10000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm import tqdm

from vitok import AE, decode_variant
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.pp import sample_tiles
from vitok import utils as tu
from vitok.metrics import MetricCalculator

import modal
from scripts.modal.modal_config import image, gpu

# Perceptual losses (imported lazily in train() to avoid issues on Modal import)
import wandb


def main():
    parser = argparse.ArgumentParser(description="Train ViTok VAE")

    # Data
    parser.add_argument("--data", type=str,
                        default="hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar",
                        help="Data source (local path or hf://repo/pattern)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)

    # Model
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant (e.g., B/1x16x64, Ld2-Ld22/1x16x64)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained model name (e.g., 'L-64') or path for finetuning. Mutually exclusive with --checkpoint.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder for decoder-only finetuning. Requires --pretrained.")

    # Training
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "muon"], help="Optimizer to use")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "warmup_exp_decay"],
                        help="LR schedule type")

    # Loss weights
    parser.add_argument("--charbonnier", type=float, default=1.0)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--ssim", type=float, default=0.1)
    parser.add_argument("--dino_perceptual", type=float, default=250.0)
    parser.add_argument("--tile_size", type=int, default=256,
                        help="Tile size for perceptual losses")
    parser.add_argument("--n_tiles", type=int, default=1,
                        help="Number of tiles per image for perceptual losses")

    # Distributed
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP2 instead of DDP")

    # Logging
    parser.add_argument("--output_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Frequency for running evaluation")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Separate validation data source (e.g., hf://ILSVRC/imagenet-1k/val/*.tar)")
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")

    args = parser.parse_args()
    train(args)


def train(args):
    """Main training function."""
    # Import perceptual losses here to avoid issues on Modal import
    from dino_perceptual import DINOPerceptual
    from torchmetrics.functional.image import structural_similarity_index_measure as SSIM

    # Validate pretrained/checkpoint args
    if args.pretrained and args.checkpoint:
        raise ValueError("--pretrained and --checkpoint are mutually exclusive. Use --pretrained for finetuning, --checkpoint for resuming.")
    if args.freeze_encoder and not args.pretrained:
        raise ValueError("--freeze_encoder requires --pretrained to be specified.")

    # Auto-infer variant from pretrained model if using pretrained
    if args.pretrained:
        from vitok.pretrained import get_pretrained_info
        try:
            _, _, variant = get_pretrained_info(args.pretrained)
            args.variant = variant
            print(f"Auto-inferred variant '{variant}' from pretrained model '{args.pretrained}'")
        except KeyError:
            # If not a known pretrained model (e.g., local file), keep user-specified variant
            pass

    # Handle compile flag (--compile is default True, --no_compile disables it)
    use_compile = args.compile and not args.no_compile

    # Setup distributed
    rank, world_size, local_rank, device, device_mesh = tu.setup_distributed(args.seed)
    dtype = torch.bfloat16
    use_fsdp = args.fsdp and world_size > 1

    # Output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_enabled = args.wandb_project and rank == 0
    if wandb_enabled:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # Create model
    if rank == 0:
        print(f"Creating AE model: {args.variant}")
    model = AE(**decode_variant(args.variant))
    model.to(device=device, dtype=dtype)

    # Load pretrained weights for finetuning (must be before compile/FSDP)
    if args.pretrained:
        from vitok.utils import load_pretrained_weights
        model = load_pretrained_weights(model, args.pretrained, device, dtype, rank, args.freeze_encoder)

    # Compile if requested (before FSDP/DDP wrapping)
    if use_compile:
        if rank == 0:
            print("Compiling model...")
        model = torch.compile(model, fullgraph=True)

    # Wrap with FSDP2 or DDP
    if world_size > 1:
        if use_fsdp:
            mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            fully_shard(model, mesh=device_mesh, mp_policy=mp)
            if rank == 0:
                print("Using FSDP2")
        else:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
            if rank == 0:
                print("Using DDP")

    # Count parameters
    model_ref = model.module if hasattr(model, 'module') else model
    n_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Optimizer

    # Separate params into decay and no_decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases, norms, embeddings
        if param.ndim <= 1 or 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if args.optimizer == "muon":
        from muon import Muon
        optimizer = Muon(model.parameters(), lr=args.lr, momentum=0.95)
    else:
        optimizer = AdamW(
            [
                {'params': decay_params, 'weight_decay': args.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=args.lr,
            betas=(0.9, 0.99),
            fused=True,
        )

    # LR Scheduler
    warmup_steps = int(args.warmup_ratio * args.steps)
    scheduler = tu.create_scheduler(
        optimizer=optimizer,
        schedule_type=args.schedule,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=warmup_steps,
    )

    # Build train_state for DCP checkpointing
    train_state = {
        "app": tu.ModelOptimizerState(model, optimizer),
        "step": 1,
        "scheduler": scheduler,
    }

    # Resume if checkpoint provided
    start_step = 0
    if args.checkpoint:
        start_step = tu.load_checkpoint(train_state, args.checkpoint, rank)
        if rank == 0:
            print(f"Resumed at step {start_step}")

    # Create dataloader (on ALL ranks)
    if rank == 0:
        print(f"Loading data from: {args.data}")

    # Build preprocessing string: 25% square crop, 75% native aspect ratio
    pp_string = (
        f"random_choice(ops=['random_resized_crop({args.max_size})', 'identity'], probs=[0.25, 0.75])|"
        f"flip|"
        f"to_tensor|"
        f"normalize(minus_one_to_one)|"
        f"resize_to_token_budget({args.patch_size}, {args.max_tokens})|"
        f"patchify({args.patch_size}, {args.max_tokens})"
    )

    loader = create_dataloader(
        source=args.data, pp=pp_string, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )

    # Perceptual losses
    dino_loss_fn = None
    if args.dino_perceptual > 0:
        dino_loss_fn = DINOPerceptual(model_size='S', target_size=args.tile_size)
        dino_loss_fn = dino_loss_fn.to(device).eval()
        if use_compile:
            dino_loss_fn = torch.compile(dino_loss_fn, fullgraph=True)

    # Evaluation metric calculator
    eval_metrics = MetricCalculator(metrics=('ssim', 'psnr'))
    eval_metrics.move_model_to_device(device)

    # Separate eval dataloader (if provided)
    eval_loader = None
    eval_iter = None
    if args.eval_data and args.eval_freq > 0:
        if rank == 0:
            print(f"Loading eval data from: {args.eval_data}")
        eval_loader = create_dataloader(
            source=args.eval_data,
            pp=pp_string,
            batch_size=args.batch_size,
            num_workers=2,
            seed=args.seed,
        )
        eval_iter = iter(eval_loader)

    # Training loop
    if rank == 0:
        print(f"\nStarting training for {args.steps} steps...")
    model.train()
    loader_iter = iter(loader)
    step = start_step

    log_metrics = {}
    log_count = 0
    t_log_start = time.perf_counter()
    data_load_times = []

    max_grid_size = args.max_size // args.patch_size
    pbar = tqdm(total=args.steps, initial=start_step, disable=rank != 0, desc="Training")

    while step < args.steps:
        data_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        data_load_time = time.perf_counter() - data_start
        data_load_times.append(data_load_time)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)
        optimizer.zero_grad(set_to_none=True)

        # Step the scheduler (handles warmup + cosine internally)
        lr = scheduler.step()
        with torch.autocast(device_type='cuda', dtype=dtype):
            decode_dict = model(batch)

        patch_mask = batch['patch_mask']
        diff = decode_dict['patches'] - batch['patches']
        diff_f32 = diff.float()
        charb_per_token = (diff_f32.pow(2) + args.charbonnier_eps**2).sqrt().mean(dim=2)
        charb_per_token = charb_per_token * patch_mask.float()
        actual_tokens = patch_mask.sum(dim=1).clamp_min(1).float()
        charb_loss = (charb_per_token.sum(dim=1) / actual_tokens).mean()

        loss = args.charbonnier * charb_loss

        # Perceptual losses on tiles
        ssim_loss = torch.tensor(0.0, device=device)
        dino_loss = torch.tensor(0.0, device=device)

        if args.ssim > 0 or args.dino_perceptual > 0:
            # Reconstruct images - recon needs gradients, ref does not
            recon_images = postprocess(
                decode_dict, output_format="minus_one_to_one",
                current_format="minus_one_to_one", unpack=False,
                patch=args.patch_size, max_grid_size=max_grid_size,
            )
            with torch.no_grad():
                ref_images = postprocess(
                    batch, output_format="minus_one_to_one",
                    current_format="minus_one_to_one", unpack=False,
                    patch=args.patch_size, max_grid_size=max_grid_size,
                )

            # Sample tiles for perceptual losses
            orig_h = batch['orig_height'].to(device)
            orig_w = batch['orig_width'].to(device)
            tiles_ref, tile_indices = sample_tiles(
                ref_images, orig_h, orig_w,
                n_tiles=args.n_tiles, tile_size=(args.tile_size, args.tile_size)
            )
            tiles_pred, _ = sample_tiles(
                recon_images, orig_h, orig_w,
                n_tiles=args.n_tiles, tile_size=(args.tile_size, args.tile_size),
                indices=tile_indices
            )

            B = tiles_ref.shape[0]
            tiles_ref = tiles_ref.reshape(B * args.n_tiles, 3, args.tile_size, args.tile_size)
            tiles_pred = tiles_pred.reshape(B * args.n_tiles, 3, args.tile_size, args.tile_size)

            # Compute perceptual losses
            with torch.autocast(device_type='cuda', dtype=dtype):
                if args.ssim > 0:
                    ssim_val = SSIM(preds=tiles_pred, target=tiles_ref, data_range=2.0)
                    ssim_loss = (1.0 - ssim_val)
                    loss = loss + args.ssim * ssim_loss

                if args.dino_perceptual > 0 and dino_loss_fn is not None:
                    dino_loss = dino_loss_fn(tiles_pred, tiles_ref).mean()
                    loss = loss + args.dino_perceptual * dino_loss

        # Backward
        loss.backward()
        #grad_norm = tu.clip_grad_norm_(grad_params, args.grad_clip, use_fsdp=use_fsdp, world_size=world_size)
        #grad_norm = 0

        optimizer.step()
        step += 1
        train_state['step'] = step
        pbar.update(1)

        # Accumulate metrics (keep as tensors to avoid GPU sync every step)
        log_metrics['loss/total'] = log_metrics.get('loss/total', 0) + loss.detach()
        log_metrics['loss/charb'] = log_metrics.get('loss/charb', 0) + charb_loss.detach()
        log_metrics['loss/ssim'] = log_metrics.get('loss/ssim', 0) + ssim_loss.detach()
        log_metrics['loss/dino'] = log_metrics.get('loss/dino', 0) + dino_loss.detach()
        log_count += 1

        # Log (only call .item() here to minimize GPU syncs)
        if step % args.log_freq == 0:
            elapsed = time.perf_counter() - t_log_start
            avg = {k: (v / log_count).item() for k, v in log_metrics.items()}
            avg['training/lr'] = lr
            #avg['training/grad_norm'] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            avg['timing/samples_per_sec'] = (args.batch_size * log_count * world_size) / elapsed
            avg['timing/step_time'] = elapsed / log_count

            # Data loading stats
            recent_data_times = data_load_times[-args.log_freq:]
            if recent_data_times:
                avg_data_load_time = sum(recent_data_times) / len(recent_data_times)
                avg['timing/data_load_ms'] = avg_data_load_time * 1000
                avg['timing/data_throughput'] = args.batch_size / avg_data_load_time

            # GPU memory stats
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
                mem_reserved = torch.cuda.max_memory_reserved() / 1e9  # GB
                avg['memory/allocated_gb'] = mem_allocated
                avg['memory/reserved_gb'] = mem_reserved
                torch.cuda.reset_peak_memory_stats()
            if step > 1:  # Skip first step (compilation)
                tokens_per_step = args.batch_size * args.max_tokens * world_size
                flops_per_step = 6 * n_params * tokens_per_step
                flops_per_sec = flops_per_step / (elapsed / log_count)
                gpu_tflops = 912e12 * world_size
                mfu = flops_per_sec / gpu_tflops * 100
                avg['timing/mfu_percent'] = mfu

            if rank == 0:
                mem_str = f" | mem: {mem_allocated:.1f}GB" if torch.cuda.is_available() else ""
                mfu_str = f" | H200 MFU: {avg.get('timing/mfu_percent', 0):.1f}%" if 'timing/mfu_percent' in avg else ""
                data_str = f" | data: {avg.get('timing/data_load_ms', 0):.1f}ms ({avg.get('timing/data_throughput', 0):.0f} imgs/s)" if 'timing/data_load_ms' in avg else ""
                print(f"Step {step}/{args.steps} | "
                      f"loss: {avg['loss/total']:.4f} | "
                      f"charb: {avg['loss/charb']:.4f} | "
                      f"ssim: {avg['loss/ssim']:.4f} | "
                      f"dino: {avg['loss/dino']:.4f} | "
                      f"lr: {lr:.2e}{mem_str}{mfu_str}{data_str}")

            if wandb_enabled:
                wandb.log(avg, step=step)

            log_metrics = {}
            log_count = 0
            t_log_start = time.perf_counter()
            data_load_times = []  # Clear to prevent unbounded memory growth

        # Evaluation
        if args.eval_freq > 0 and step % args.eval_freq == 0:
            model.eval()
            eval_metrics.reset()

            # Use separate eval loader if provided, otherwise use training loader
            use_eval_iter = eval_iter if eval_iter is not None else loader_iter
            n_eval_batches = 10
            all_ref = []
            all_recon = []

            with torch.no_grad():
                for _ in range(n_eval_batches):
                    try:
                        eval_batch = next(use_eval_iter)
                    except StopIteration:
                        use_eval_iter = iter(eval_loader) if eval_loader else iter(loader)
                        eval_batch = next(use_eval_iter)

                    eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
                    if 'patches' in eval_batch:
                        eval_batch['patches'] = eval_batch['patches'].to(dtype)

                    with torch.autocast(device_type='cuda', dtype=dtype):
                        eval_out = model(eval_batch)

                    recon = postprocess(
                        eval_out, output_format="minus_one_to_one",
                        current_format="minus_one_to_one", unpack=True,
                        patch=args.patch_size, max_grid_size=max_grid_size,
                    )
                    ref = postprocess(
                        eval_batch, output_format="minus_one_to_one",
                        current_format="minus_one_to_one", unpack=True,
                        patch=args.patch_size, max_grid_size=max_grid_size,
                    )

                    if isinstance(ref, list):
                        all_ref.extend([x.cpu() for x in ref])
                        all_recon.extend([x.cpu() for x in recon])
                    else:
                        all_ref.append(ref.cpu())
                        all_recon.append(recon.cpu())

            eval_metrics.update(all_ref, all_recon)
            eval_stats = eval_metrics.gather()

            if rank == 0:
                eval_source = "val" if args.eval_data else "train"
                print(f"[Eval @ {step}] ({eval_source}) SSIM: {eval_stats.get('ssim', 0):.4f} | PSNR: {eval_stats.get('psnr', 0):.2f}")

            if wandb_enabled:
                wandb.log({f"eval/{k}": v for k, v in eval_stats.items()}, step=step)

            model.train()

        # Save checkpoint
        if step % args.save_freq == 0:
            tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)
            if rank == 0:
                print(f"Saved checkpoint at step {step}")

    pbar.close()

    # Final save
    tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)
    if rank == 0:
        print(f"\nTraining complete! Final checkpoint saved.")

    if wandb_enabled:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Modal support: modal run scripts/train_vae.py --steps 100000
# =============================================================================
app = modal.App("vitok-train")


@app.function(image=image, **gpu("A100:8", timeout=86400))
def run_training_remote(
    data: str = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar",
    variant: str = "Ld2-Ld22/1x16x64",
    batch_size: int = 32,
    steps: int = 100000,
    lr: float = 3e-4,
    pretrained: str = None,
    freeze_encoder: bool = False,
    wandb_project: str = None,
    wandb_name: str = None,
    output_dir: str = "/output/checkpoints",
):
    """Run distributed training on Modal (8x A100 with FSDP)."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=8", "scripts/train_vae.py", "--fsdp",
        "--data", data,
        "--variant", variant,
        "--batch-size", str(batch_size),
        "--steps", str(steps),
        "--lr", str(lr),
        "--output-dir", output_dir,
    ]

    if pretrained:
        cmd.extend(["--pretrained", pretrained])
    if freeze_encoder:
        cmd.append("--freeze-encoder")
    if wandb_project:
        cmd.extend(["--wandb-project", wandb_project])
    if wandb_name:
        cmd.extend(["--wandb-name", wandb_name])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def modal_main(
    data: str = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..1023}.tar",
    variant: str = "Ld2-Ld22/1x16x64",
    batch_size: int = 32,
    steps: int = 100000,
    lr: float = 3e-4,
    pretrained: str = None,
    freeze_encoder: bool = False,
    wandb_project: str = None,
    wandb_name: str = None,
):
    """Modal entrypoint for VAE training.

    Examples:
        # Default training (ImageNet-22k, 8x A100)
        modal run scripts/train_vae.py --steps 100000 --wandb-project vitok

        # Custom data
        modal run scripts/train_vae.py --data hf://ILSVRC/imagenet-1k/train-{00000..01023}.tar

        # Finetune from pretrained
        modal run scripts/train_vae.py --pretrained 350M-f16x64 --freeze-encoder --steps 10000
    """
    print(f"Starting training on Modal (8x A100)...")
    print(f"  Data: {data}")
    print(f"  Variant: {variant}")
    print(f"  Steps: {steps}")

    run_training_remote.remote(
        data=data,
        variant=variant,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    print("Training complete! Checkpoints saved to Modal volume: vitok-output:/checkpoints")


if __name__ == "__main__":
    main()
