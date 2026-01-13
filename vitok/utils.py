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
    model: Optional[nn.Module] = None,
):
    """Save training checkpoint using DCP.

    Args:
        train_state: Dict containing 'app' (ModelOptimizerState), 'step', 'scheduler', etc.
        checkpoint_root: Root directory for checkpoints
        model: Optional model to also save as safetensors (gathered to rank 0)
    """
    last_path = os.path.join(checkpoint_root, "last")
    os.makedirs(last_path, exist_ok=True)
    dcp.save(state_dict=train_state, checkpoint_id=last_path)
    if model is not None:
        export_safetensors(model, last_path)

def load_checkpoint(train_state: Dict[str, Any], checkpoint_path: str) -> int:
    """Load training checkpoint using DCP. Returns step number."""
    dcp.load(state_dict=train_state, checkpoint_id=checkpoint_path)
    return train_state.get('step', 0)


def export_safetensors(model: nn.Module, save_dir: str):
    """Export model weights as safetensors (gathered to rank 0)."""
    import itertools
    state_dict = {}
    rank = dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            if rank == 0:
                state_dict[name] = param.detach().cpu()

    if rank == 0:
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


class BaseScheduler(Stateful):
    """Base LR scheduler with warmup."""

    def __init__(self, optimizer, warmup_steps: int, max_lr: float, start_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_lr(self) -> float:
        raise NotImplementedError

    def set_step(self, step: int):
        self.current_step = step

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


class CosineScheduler(BaseScheduler):
    """Cosine annealing LR scheduler with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float, min_lr: float = 1e-6, start_lr: float = 1e-7):
        super().__init__(optimizer, warmup_steps, max_lr, start_lr)
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            return self.start_lr + (self.max_lr - self.start_lr) * (self.current_step / max(1, self.warmup_steps))
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


class LinearScheduler(BaseScheduler):
    """Linear warmup followed by constant LR."""

    def get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            return self.start_lr + (self.max_lr - self.start_lr) * (self.current_step / max(1, self.warmup_steps))
        return self.max_lr


class ExponentialDecayScheduler(BaseScheduler):
    """Linear warmup followed by exponential decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float, final_lr: float = 1e-6, start_lr: float = 1e-7):
        super().__init__(optimizer, warmup_steps, max_lr, start_lr)
        self.total_steps = total_steps
        self.final_lr = final_lr
        self.decay_rate = (final_lr / max_lr) ** (1.0 / max(1, total_steps - warmup_steps))

    def get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            return self.start_lr + (self.max_lr - self.start_lr) * (self.current_step / max(1, self.warmup_steps))
        decay_step = self.current_step - self.warmup_steps
        return max(self.max_lr * (self.decay_rate ** decay_step), self.final_lr)


def create_scheduler(
    optimizer,
    schedule_type: str,
    steps: int,
    lr: float,
    warmup_steps: Optional[int] = None,
    start_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
):
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        schedule_type: One of 'cosine', 'linear', 'warmup_exp_decay'
        steps: Total training steps
        lr: Peak learning rate
        warmup_steps: Warmup steps (defaults to 5% of total)
        start_lr: Starting LR for warmup (defaults to 1e-7)
        final_lr: Final LR for decay schedules (defaults to 1e-6)

    Returns:
        Scheduler instance
    """
    if warmup_steps is None:
        warmup_steps = int(0.05 * steps)
    if start_lr is None:
        start_lr = 1e-7
    if final_lr is None:
        final_lr = 1e-5

    if schedule_type == "cosine":
        return CosineScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=steps,
            max_lr=lr, min_lr=final_lr, start_lr=start_lr
        )
    elif schedule_type == "linear":
        return LinearScheduler(
            optimizer, warmup_steps=warmup_steps,
            max_lr=lr, start_lr=start_lr
        )
    elif schedule_type == "warmup_exp_decay":
        return ExponentialDecayScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=steps,
            max_lr=lr, final_lr=final_lr, start_lr=start_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")
