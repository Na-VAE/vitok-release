#!/usr/bin/env python
"""DiT training script with flow matching and UniPC sampling.

Example usage:
    # Single GPU
    python scripts/train_dit.py --config configs/dit_imagenet.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/train_dit.py --config configs/dit_imagenet.py

    # With FSDP
    torchrun --nproc_per_node=8 scripts/train_dit.py --config configs/dit_imagenet.py --fsdp
"""

import argparse
import os
import random
import time
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from vitok import AEConfig, create_ae, load_ae
from vitok import DiTConfig, create_dit, load_dit
from vitok import StreamingWebDatasetConfig, create_streaming_dataloader
from vitok.diffusion import FlowUniPCMultistepScheduler
from vitok.datasets.io import postprocess_images
from vitok.utils.weights import load_weights


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    ae_variant: str = "Ld2-Ld22/1x16x64"
    ae_checkpoint: Optional[str] = None
    dit_variant: str = "L/256"
    dit_checkpoint: Optional[str] = None
    num_classes: int = 1000

    # Data
    data_paths: List[str] = field(default_factory=lambda: [])
    hf_repo: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    max_tokens: int = 256
    image_size: int = 256

    # Training
    steps: int = 400000
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    cfg_dropout: float = 0.1

    # EMA
    ema_decay: float = 0.9999
    ema_start_step: int = 5000

    # Logging & Checkpointing
    log_freq: int = 100
    sample_freq: int = 5000
    save_freq: int = 10000
    output_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # System
    seed: int = 42
    compile: bool = False
    fsdp: bool = False
    bf16: bool = True


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, local_rank, device


def requires_grad(model: nn.Module, flag: bool = True):
    """Set requires_grad for all parameters."""
    for p in model.parameters():
        p.requires_grad = flag


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    """Update EMA model parameters."""
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, param in model.named_parameters():
            if name in ema_params:
                ema_params[name].lerp_(param.data, 1 - decay)


def get_cosine_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create cosine LR schedule with warmup."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model,
    ema,
    optimizer,
    scheduler,
    step: int,
    output_dir: str,
    rank: int,
):
    """Save training checkpoint."""
    if rank != 0:
        return

    checkpoint_dir = Path(output_dir) / f"step_{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get state dict (handle FSDP/DDP)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    if hasattr(ema, 'module'):
        ema_state = ema.module.state_dict()
    else:
        ema_state = ema.state_dict()

    from safetensors.torch import save_file

    save_file(model_state, checkpoint_dir / "model.safetensors")
    save_file(ema_state, checkpoint_dir / "ema.safetensors")

    # Save training state
    torch.save({
        'step': step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, checkpoint_dir / "train_state.pt")

    print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train DiT with flow matching")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--ae_checkpoint", type=str, help="Path to AE checkpoint")
    parser.add_argument("--dit_checkpoint", type=str, help="Path to DiT checkpoint (resume)")
    parser.add_argument("--data_paths", type=str, nargs="+", help="Paths to data directories")
    parser.add_argument("--hf_repo", type=str, help="HuggingFace dataset repo")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    config = TrainConfig()
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        if hasattr(cfg_module, 'config'):
            for k, v in vars(cfg_module.config).items():
                if not k.startswith('_'):
                    setattr(config, k, v)

    # Override with command line args
    for k, v in vars(args).items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()

    # Set seed
    torch.manual_seed(config.seed + rank)
    random.seed(config.seed + rank)

    # Dtype
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float32

    def autocast_ctx():
        if config.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # Create AE (frozen encoder)
    ae_config = AEConfig(
        variant=config.ae_variant,
        variational=True,
    )
    ae = load_ae(config.ae_checkpoint, ae_config, device=device, dtype=dtype)
    ae.eval()
    requires_grad(ae, False)

    # Get code width from AE
    code_width = ae.encoder_width if hasattr(ae, 'encoder_width') else 64

    # Create DiT
    dit_config = DiTConfig(
        variant=config.dit_variant,
        code_width=code_width,
        num_classes=config.num_classes,
    )
    dit = create_dit(dit_config)
    dit.to(device=device, dtype=dtype)

    # EMA
    ema = deepcopy(dit)
    requires_grad(ema, False)
    ema.eval()

    # Load checkpoint if resuming
    start_step = 0
    if config.dit_checkpoint:
        load_weights(dit, config.dit_checkpoint, strict=False)
        load_weights(ema, config.dit_checkpoint, strict=False)
        # Try to load training state
        train_state_path = Path(config.dit_checkpoint).parent / "train_state.pt"
        if train_state_path.exists():
            state = torch.load(train_state_path, map_location='cpu')
            start_step = state.get('step', 0)
            print(f"Resuming from step {start_step}")

    n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    if rank == 0:
        print(f"DiT parameters: {n_params / 1e6:.2f}M")

    # Compile before FSDP
    if config.compile:
        if rank == 0:
            print("Compiling models...")
        dit = torch.compile(dit, fullgraph=True)
        ema = torch.compile(ema, fullgraph=True)
        ae = torch.compile(ae, fullgraph=True)

    # Wrap with FSDP or DDP
    if config.fsdp and world_size > 1:
        mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
        fully_shard(dit, mp_policy=mp)
        fully_shard(ema, mp_policy=MixedPrecisionPolicy(param_dtype=dtype))
    elif world_size > 1:
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[local_rank], find_unused_parameters=False
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )

    # Scheduler
    warmup_steps = int(config.warmup_ratio * config.steps)
    lr_scheduler = get_cosine_schedule(optimizer, warmup_steps, config.steps)

    # Restore optimizer state if resuming
    if config.dit_checkpoint:
        train_state_path = Path(config.dit_checkpoint).parent / "train_state.pt"
        if train_state_path.exists():
            state = torch.load(train_state_path, map_location='cpu')
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
            if 'scheduler' in state:
                lr_scheduler.load_state_dict(state['scheduler'])

    # Scheduler for training and sampling (flow matching with velocity prediction)
    diffusion_scheduler = FlowUniPCMultistepScheduler(thresholding=False)
    diffusion_scheduler.set_timesteps(1000, device=device)
    eval_scheduler = FlowUniPCMultistepScheduler(thresholding=False)

    # Dataloader
    if config.hf_repo:
        from vitok.datasets.webdataset import HFWebDataset
        from vitok.transforms import TransformCfg, build_transform
        from vitok.transforms.collate import patch_collate_fn

        transform_cfg = TransformCfg(
            train=True,
            patch_size=16,
            max_tokens=config.max_tokens,
            max_size=config.image_size,
        )
        transform = build_transform(transform_cfg)

        dataset = HFWebDataset(
            hf_repo=config.hf_repo,
            transform=transform,
            collate_fn=patch_collate_fn,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            return_labels=True,
        )
        dataloader = dataset.create_dataloader()
    elif config.data_paths:
        data_config = StreamingWebDatasetConfig(
            bucket_paths=config.data_paths,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_tokens=config.max_tokens,
            seed=config.seed,
            return_labels=True,
        )
        dataloader = create_streaming_dataloader(data_config)
    else:
        raise ValueError("Must provide either data_paths or hf_repo")

    data_iter = iter(dataloader)

    # WandB
    if rank == 0 and config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=vars(config),
        )

    # Training loop
    dit.train()
    step = start_step

    if rank == 0:
        print(f"Starting training from step {step}")
        print(f"Total steps: {config.steps}")

    while step < config.steps:
        step_start = time.time()

        # Get batch
        try:
            patch_dict, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            patch_dict, labels = next(data_iter)

        # Move to device
        patch_dict = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in patch_dict.items()
        }
        labels = labels.to(device)

        # Encode with frozen AE
        with torch.no_grad():
            with autocast_ctx():
                patch_dict['patches'] = patch_dict['patches'].to(dtype)
                posterior = ae.encode(patch_dict)
                z = posterior['posterior'].sample() if hasattr(posterior.get('posterior', {}), 'sample') else posterior['z']
                z = z.float()

        # CFG dropout
        context = labels.clone()
        dropout_mask = torch.rand(labels.shape[0], device=device) < config.cfg_dropout
        context[dropout_mask] = config.num_classes  # Null class

        # Sample timesteps and add noise (flow matching)
        noise = torch.randn_like(z)
        t_idx = torch.randint(0, 1000, (z.shape[0],), device=device)
        noisy_z = diffusion_scheduler.add_noise(z, noise, t_idx)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            v_pred = dit({
                "z": noisy_z,
                "t": t_idx.float(),
                "context": context,
                "yidx": patch_dict.get("yidx"),
                "xidx": patch_dict.get("xidx"),
            })

        # Flow matching loss: predict velocity v = noise - z
        velocity_target = noise - z
        loss = (v_pred.float() - velocity_target).pow(2).mean()

        # Backward
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), config.grad_clip)
        else:
            grad_norm = 0.0

        optimizer.step()
        lr_scheduler.step()

        step += 1

        # EMA update
        if step < config.ema_start_step:
            update_ema(ema, dit, decay=0.0)  # Copy weights
        else:
            update_ema(ema, dit, decay=config.ema_decay)

        # Logging
        if step % config.log_freq == 0:
            torch.cuda.synchronize()
            step_time = time.time() - step_start
            samples_per_sec = config.batch_size / step_time

            if rank == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"Step {step}/{config.steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad: {grad_norm:.2f} | "
                    f"Samples/s: {samples_per_sec:.1f}"
                )

                if config.wandb_project:
                    import wandb
                    wandb.log({
                        "loss": loss.item(),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "samples_per_sec": samples_per_sec,
                        "step": step,
                    })

        # Save checkpoint
        if step % config.save_freq == 0:
            save_checkpoint(dit, ema, optimizer, lr_scheduler, step, config.output_dir, rank)

        # Sample visualization
        if step % config.sample_freq == 0 and rank == 0:
            dit.eval()
            with torch.no_grad():
                # Generate samples
                sample_labels = torch.arange(min(16, config.num_classes), device=device)
                sample_labels_null = torch.full_like(sample_labels, config.num_classes)
                latents = torch.randn(len(sample_labels), z.shape[1], z.shape[2], device=device)

                # Sampling with UniPC
                eval_scheduler.set_timesteps(50)
                for i, t in enumerate(eval_scheduler.timesteps):
                    t_batch = t.expand(latents.shape[0])

                    # CFG
                    x_in = torch.cat([latents, latents], dim=0)
                    t_in = torch.cat([t_batch, t_batch], dim=0)
                    y_in = torch.cat([sample_labels_null, sample_labels], dim=0)

                    with autocast_ctx():
                        out = ema({"z": x_in, "t": t_in, "context": y_in})

                    uncond, cond = out.chunk(2, dim=0)
                    v_pred = uncond + 4.0 * (cond - uncond)

                    latents = eval_scheduler.step(v_pred.float(), t, latents, return_dict=False)[0]

                # Decode with AE
                # Create minimal patch dict for decoding
                L = latents.shape[1]
                side = int(L ** 0.5)
                y, x = torch.meshgrid(
                    torch.arange(side, device=device),
                    torch.arange(side, device=device),
                    indexing='ij'
                )
                decode_dict = {
                    'z': latents.to(dtype),
                    'ptype': torch.ones(len(sample_labels), L, dtype=torch.bool, device=device),
                    'yidx': y.flatten().unsqueeze(0).expand(len(sample_labels), -1),
                    'xidx': x.flatten().unsqueeze(0).expand(len(sample_labels), -1),
                    'original_height': torch.full((len(sample_labels),), config.image_size, device=device),
                    'original_width': torch.full((len(sample_labels),), config.image_size, device=device),
                }

                with autocast_ctx():
                    decoded = ae.decode(decode_dict)

                images = postprocess_images(
                    decoded,
                    output_format="zero_to_one",
                    return_type="tensor",
                    unpack=True,
                    spatial_stride=16,
                )

                if config.wandb_project:
                    import wandb
                    wandb.log({
                        f"samples/class_{i}": wandb.Image(img.cpu())
                        for i, img in enumerate(images[:8])
                    }, step=step)

            dit.train()

    # Final checkpoint
    save_checkpoint(dit, ema, optimizer, lr_scheduler, step, config.output_dir, rank)

    if rank == 0:
        print("Training complete!")
        if config.wandb_project:
            import wandb
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
