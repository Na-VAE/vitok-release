#!/usr/bin/env python
"""DiT training script with flow matching and UniPC sampling.

Example usage:
    # Single GPU
    python scripts/train_dit.py --config configs/dit_imagenet.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/train_dit.py --config configs/dit_imagenet.py

    # With FSDP2
    torchrun --nproc_per_node=8 scripts/train_dit.py --config configs/dit_imagenet.py --fsdp
"""

import argparse
import math
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
from torch.nn.parallel import DistributedDataParallel as DDP

from vitok import AE, decode_variant, DiT, decode_dit_variant
from vitok import create_dataloader
from vitok.unipc import FlowUniPCMultistepScheduler
from vitok.naflex_io import postprocess_images
from safetensors.torch import load_file
from vitok import utils as tu


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
    marked_freq: int = 50000
    output_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # System
    seed: int = 42
    compile: bool = False
    fsdp: bool = False
    bf16: bool = True


def requires_grad(model: nn.Module, flag: bool = True):
    """Set requires_grad for all parameters."""
    for p in model.parameters():
        p.requires_grad = flag


def main():
    parser = argparse.ArgumentParser(description="Train DiT with flow matching")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP2")
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
    rank, world_size, local_rank, device, device_mesh = tu.setup_distributed(config.seed)
    use_fsdp = config.fsdp and world_size > 1
    dtype = torch.bfloat16 if config.bf16 else torch.float32

    def autocast_ctx():
        if config.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # Create AE (frozen encoder)
    ae_params = decode_variant(config.ae_variant)
    ae = AE(**ae_params)
    if config.ae_checkpoint:
        ae.load_state_dict(load_file(config.ae_checkpoint))
    ae.to(device=device, dtype=dtype)
    ae.eval()
    requires_grad(ae, False)

    # Get code width from AE
    code_width = ae_params.get('channels_per_token', 64)

    # Create DiT
    dit_params = decode_dit_variant(config.dit_variant)
    dit = DiT(**dit_params, code_width=code_width, text_dim=config.num_classes)
    dit.to(device=device, dtype=dtype)

    # EMA
    ema = deepcopy(dit)
    requires_grad(ema, False)
    ema.eval()

    # Load checkpoint if resuming
    start_step = 0
    if config.dit_checkpoint:
        state = load_file(config.dit_checkpoint)
        dit.load_state_dict(state, strict=False)
        ema.load_state_dict(state, strict=False)
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

    # Wrap with FSDP2 or DDP
    if world_size > 1:
        if use_fsdp:
            mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            fully_shard(dit, mesh=device_mesh, mp_policy=mp)
            fully_shard(ema, mesh=device_mesh, mp_policy=MixedPrecisionPolicy(param_dtype=dtype))
            if rank == 0:
                print("Using FSDP2")
        else:
            dit = DDP(dit, device_ids=[local_rank], find_unused_parameters=False)
            if rank == 0:
                print("Using DDP")

    # Optimizer
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )

    # Build train_state for DCP checkpointing
    train_state = {
        "app": tu.ModelOptimizerState(dit, optimizer),
        "step": start_step,
    }

    # Restore optimizer state if resuming with DCP
    if config.dit_checkpoint:
        try:
            tu.load_checkpoint(train_state, config.dit_checkpoint, rank)
            start_step = train_state.get('step', start_step)
        except Exception:
            pass  # Fall back to manual loading above

    # Scheduler for training and sampling (flow matching with velocity prediction)
    diffusion_scheduler = FlowUniPCMultistepScheduler(thresholding=False)
    diffusion_scheduler.set_timesteps(1000, device=device)
    eval_scheduler = FlowUniPCMultistepScheduler(thresholding=False)

    # Dataloader
    pp_str = f"random_resized_crop({config.image_size})|flip|to_tensor|normalize(minus_one_to_one)|patchify({config.image_size}, 16, {config.max_tokens})"

    if config.hf_repo:
        source = f"hf://{config.hf_repo}/*.tar"
    elif config.data_paths:
        source = config.data_paths[0]
    else:
        raise ValueError("Must provide either data_paths or hf_repo")

    dataloader = create_dataloader(
        source=source,
        pp=pp_str,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed + start_step,
        return_labels=True,
    )

    data_iter = iter(dataloader)

    # WandB
    if rank == 0 and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_name, config=vars(config))

    # Output directory
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    dit.train()
    step = start_step
    warmup_steps = int(config.warmup_ratio * config.steps)
    grad_params = [p for p in dit.parameters() if p.requires_grad]

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

        # Learning rate schedule
        if step < warmup_steps:
            lr = config.lr * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, config.steps - warmup_steps)
            lr = config.lr * 0.1 + config.lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

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
        grad_norm = tu.clip_grad_norm_(grad_params, config.grad_clip, use_fsdp=use_fsdp, world_size=world_size)

        optimizer.step()
        step += 1
        train_state['step'] = step

        # EMA update
        if step < config.ema_start_step:
            tu.update_ema(ema, dit, decay=0.0)  # Copy weights
        else:
            tu.update_ema(ema, dit, decay=config.ema_decay)

        # Logging
        if step % config.log_freq == 0:
            torch.cuda.synchronize()
            step_time = time.time() - step_start
            samples_per_sec = config.batch_size / step_time

            if rank == 0:
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
                        "grad_norm": float(grad_norm),
                        "samples_per_sec": samples_per_sec,
                        "step": step,
                    })

        # Save checkpoint
        if step % config.save_freq == 0:
            tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)
            if rank == 0:
                print(f"Saved checkpoint at step {step}")

        # Marked checkpoint
        if config.marked_freq > 0 and step % config.marked_freq == 0:
            tu.save_marked_checkpoint(train_state, str(output_dir), step, rank)
            if rank == 0:
                print(f"Saved marked checkpoint at step {step}")

        # Sample visualization
        if step % config.sample_freq == 0 and rank == 0:
            dit.eval()
            with torch.no_grad():
                sample_labels = torch.arange(min(16, config.num_classes), device=device)
                sample_labels_null = torch.full_like(sample_labels, config.num_classes)
                latents = torch.randn(len(sample_labels), z.shape[1], z.shape[2], device=device)

                eval_scheduler.set_timesteps(50)
                for t in eval_scheduler.timesteps:
                    t_batch = t.expand(latents.shape[0])

                    x_in = torch.cat([latents, latents], dim=0)
                    t_in = torch.cat([t_batch, t_batch], dim=0)
                    y_in = torch.cat([sample_labels_null, sample_labels], dim=0)

                    with autocast_ctx():
                        out = ema({"z": x_in, "t": t_in, "context": y_in})

                    uncond, cond = out.chunk(2, dim=0)
                    v_pred = uncond + 4.0 * (cond - uncond)
                    latents = eval_scheduler.step(v_pred.float(), t, latents, return_dict=False)[0]

                # Decode with AE
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
    tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)

    if rank == 0:
        print("Training complete!")
        if config.wandb_project:
            import wandb
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
