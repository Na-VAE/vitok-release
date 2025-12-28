"""Training utilities shared across VAE and DiT training scripts.

Provides distributed setup, DCP checkpointing with FSDP2 support, and EMA.
"""

import os
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed._tensor import DTensor
from safetensors.torch import save_file


class ModelOptimizerState(Stateful):
    """Wrapper for proper FSDP2 model + optimizer checkpoint handling via DCP.

    See: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def setup_distributed(seed: int = 42) -> Tuple[int, int, int, torch.device, Optional[DeviceMesh]]:
    """Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, local_rank, device, device_mesh)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=device,
        )
        if rank == 0:
            print(f"Initialized process group: rank {rank}, world size {world_size}")
        torch.cuda.empty_cache()
        dist.barrier()

    # Seeds
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # Performance
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)

    # Device mesh for FSDP2
    device_mesh = init_device_mesh("cuda", (world_size,)) if world_size > 1 else None

    return rank, world_size, local_rank, device, device_mesh


def save_checkpoint(
    train_state: Dict[str, Any],
    checkpoint_root: str,
    step: int,
    rank: int,
    world_size: int,
    save_safetensors: bool = True,
):
    """Save training checkpoint using DCP.

    Args:
        train_state: Dict containing 'app' (ModelOptimizerState), 'step', 'scheduler', etc.
        checkpoint_root: Root directory for checkpoints
        step: Current training step
        rank: Process rank
        world_size: World size
        save_safetensors: Also export model weights as safetensors
    """
    train_state['step'] = step
    last_path = os.path.join(checkpoint_root, "last")
    os.makedirs(last_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    dcp.save(state_dict=train_state, checkpoint_id=last_path)

    if dist.is_initialized():
        dist.barrier()

    if save_safetensors:
        _save_safetensors(train_state, last_path, rank)

    if dist.is_initialized():
        dist.barrier()


def save_marked_checkpoint(
    train_state: Dict[str, Any],
    checkpoint_root: str,
    step: int,
    rank: int,
):
    """Save a marked checkpoint at a specific step (for periodic saves)."""
    marked_path = os.path.join(checkpoint_root, "marked", f"{step:07d}")
    os.makedirs(marked_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    _save_safetensors(train_state, marked_path, rank)

    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(train_state: Dict[str, Any], checkpoint_path: str, rank: int = 0) -> int:
    """Load training checkpoint using DCP. Returns step number."""
    ckpt_dir = _find_checkpoint_dir(checkpoint_path)
    if rank == 0:
        print(f"Loading checkpoint from: {ckpt_dir}")

    dcp.load(state_dict=train_state, checkpoint_id=ckpt_dir)

    if rank == 0:
        print("Checkpoint loaded successfully.")

    return train_state.get('step', 0)


def _find_checkpoint_dir(path: str) -> str:
    """Find the actual checkpoint directory."""
    path = os.path.abspath(os.path.expanduser(path))

    if os.path.exists(os.path.join(path, ".metadata")):
        return path

    last_path = os.path.join(path, "last")
    if os.path.exists(os.path.join(last_path, ".metadata")):
        return last_path

    marked_root = os.path.join(path, "marked")
    if os.path.isdir(marked_root):
        dirs = sorted([d for d in os.listdir(marked_root) if d.isdigit()], reverse=True)
        for d in dirs:
            p = os.path.join(marked_root, d)
            if os.path.exists(os.path.join(p, ".metadata")) or os.path.exists(os.path.join(p, "model.safetensors")):
                return p

    raise FileNotFoundError(f"No valid checkpoint found at {path}")


def _save_safetensors(train_state: Dict[str, Any], save_dir: str, rank: int):
    """Save model weights as safetensors (gathered to rank 0)."""
    if 'app' not in train_state:
        return

    model = train_state['app'].model
    if hasattr(model, 'module'):
        model = model.module

    state_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                t = param.full_tensor()
                if rank == 0:
                    state_dict[name] = t.cpu()
            elif rank == 0:
                state_dict[name] = param.detach().cpu()

        for name, buffer in model.named_buffers():
            if isinstance(buffer, DTensor):
                t = buffer.full_tensor()
                if rank == 0:
                    state_dict[name] = t.cpu()
            elif rank == 0:
                state_dict[name] = buffer.detach().cpu()

    if rank == 0 and state_dict:
        save_file(state_dict, os.path.join(save_dir, 'model.safetensors'))


@torch.inference_mode()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999):
    """Update EMA model weights in float32 for numerical stability."""
    def _to_local(t):
        return t.to_local() if hasattr(t, 'to_local') else t

    src = model.module if hasattr(model, "module") else model
    tgt = ema_model.module if hasattr(ema_model, "module") else ema_model

    tgt_params = [_to_local(p) for p in tgt.parameters()]
    src_params = [_to_local(p) for p in src.parameters()]

    if not tgt_params:
        return

    target_dtype = tgt_params[0].dtype

    if decay == 0.0:
        torch._foreach_copy_(tgt_params, src_params)
    else:
        tgt_f32 = [p.float() for p in tgt_params]
        src_f32 = [p.float() for p in src_params]
        torch._foreach_lerp_(tgt_f32, src_f32, 1.0 - decay)
        tgt_casted = [p.to(dtype=target_dtype) for p in tgt_f32]
        torch._foreach_copy_(tgt_params, tgt_casted)


def clip_grad_norm_(parameters, max_norm: float, use_fsdp: bool = False, world_size: int = 1) -> torch.Tensor:
    """Clip gradient norm with FSDP2-compatible local clipping."""
    if max_norm <= 0:
        return torch.tensor(0.0)

    def _to_local(t):
        return t.to_local() if hasattr(t, 'to_local') else t

    params = parameters if isinstance(parameters, list) else list(parameters)
    grads = [p.grad for p in params if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)

    if use_fsdp and world_size > 1:
        local_max_norm = max_norm / (world_size ** 0.5)
        local_grads = [_to_local(g) for g in grads]
        norms = torch._foreach_norm(local_grads, ord=2)
        total_norm = torch.stack(norms).norm(2)

        if total_norm > local_max_norm:
            clip_coef = local_max_norm / (total_norm + 1e-6)
            torch._foreach_mul_(local_grads, clip_coef)

        return total_norm
    else:
        return torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)
