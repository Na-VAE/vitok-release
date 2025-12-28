"""NaFlex (Flexible Resolution) patchification wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from PIL import Image

from vitok.transforms import TransformCfg, build_transform, patch_collate_fn


@dataclass(frozen=True)
class NaFlexConfig:
    """Configuration for NaFlex patchification.

    Args:
        patch_size: Size of each patch in pixels (default: 16)
        max_tokens: Maximum sequence length (default: 256)
        train_max_grid_size: Max grid size for training (default: 48)
        posemb_max_grid_size: Max grid size for positional embeddings (default: 256)
        min_size: Minimum image dimension (default: 224)
        max_size: Maximum image dimension (default: 1024)
        normalise: Normalization mode ("minus_one_to_one" or "0_to_1")
        train: Whether to apply training augmentations
        use_naflex_posemb: Enable NaFlex positional embeddings
        square_crop_prob: Probability of forcing square crop
        square_crop_sizes: Sizes for square crops
        true_random_crop_prob: Probability of true random crop
    """

    patch_size: int = 16
    max_tokens: int = 256
    train_max_grid_size: int = 48
    posemb_max_grid_size: int = 256
    min_size: int = 224
    max_size: int = 1024
    normalise: str = "minus_one_to_one"
    train: bool = False
    use_naflex_posemb: bool = True
    square_crop_prob: float = 0.25
    square_crop_sizes: Sequence[int] = (256, 512)
    true_random_crop_prob: float = 0.25


def build_naflex_transform(config: NaFlexConfig):
    """Build the NaFlex transform pipeline for PIL inputs.

    Args:
        config: NaFlexConfig instance

    Returns:
        Transform function that converts PIL Image to patch dict
    """
    cfg = TransformCfg(
        train=config.train,
        patch_size=config.patch_size,
        max_tokens=config.max_tokens,
        train_max_grid_size=config.train_max_grid_size,
        posemb_max_grid_size=config.posemb_max_grid_size,
        min_size=config.min_size,
        max_size=config.max_size,
        normalise=config.normalise,
        augmentation_strategy="naflex",
        use_naflex_posemb=config.use_naflex_posemb,
        square_crop_prob=config.square_crop_prob,
        square_crop_sizes=tuple(config.square_crop_sizes),
        true_random_crop_prob=config.true_random_crop_prob,
    )
    return build_transform(cfg, input_format="pil")


def naflex_batch(
    images: Image.Image | torch.Tensor | Iterable[Image.Image | torch.Tensor],
    config: NaFlexConfig,
    device: Optional[torch.device | str] = None,
):
    """Create a batched NaFlex patch dict from images.

    Args:
        images: Single image or iterable of images (PIL or Tensor)
        config: NaFlexConfig instance
        device: Optional target device for tensors

    Returns:
        Batched patch dictionary with keys:
            - patches: [B, N, D] flattened patch pixels
            - ptype: [B, N] patch type mask (True for valid patches)
            - yidx: [B, N] vertical patch indices
            - xidx: [B, N] horizontal patch indices
            - attention_mask: [B, N, N] attention mask
            - original_height: [B] original image heights
            - original_width: [B] original image widths
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    transform = build_naflex_transform(config)
    patch_dicts = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = _tensor_to_pil(img, config.normalise)
        patch_dicts.append(transform(img))

    batch = patch_collate_fn([(d, 0) for d in patch_dicts])[0]
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def _tensor_to_pil(img: torch.Tensor, normalise: str) -> Image.Image:
    """Convert tensor to PIL Image."""
    if img.ndim == 4:
        img = img.squeeze(0)
    if normalise == "minus_one_to_one":
        img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255).to(torch.uint8)
    return Image.fromarray(img.cpu().numpy().transpose(1, 2, 0))


__all__ = [
    "NaFlexConfig",
    "build_naflex_transform",
    "naflex_batch",
]
