"""NaFlex (Flexible Resolution) patchification and image I/O utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from vitok.transforms import (
    TransformCfg,
    build_transform,
    patch_collate_fn,
    unpatchify,
    unpack_images,
)


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


def preprocess_images(
    images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]],
    spatial_stride: int = 16,
    max_size: int = 512,
    max_seq_len: Optional[int] = 4096,
    normalise: str = "minus_one_to_one",
    train_max_grid_size: int = 64,
    posemb_max_grid_size: int = 64,
    device: str = "cuda",
):
    """Preprocess images into the patch dictionary expected by VAE/DiT.

    Args:
        images: PIL Image(s) or tensor(s)
        spatial_stride: Patch size (default: 16)
        max_size: Maximum image dimension
        max_seq_len: Maximum sequence length
        normalise: Normalization mode
        train_max_grid_size: Max grid size for training
        posemb_max_grid_size: Max grid size for positional embeddings
        device: Target device

    Returns:
        Batched patch dictionary
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    cfg = TransformCfg(
        train=False,
        patch_size=spatial_stride,
        max_size=max_size,
        normalise=normalise,
        train_max_grid_size=train_max_grid_size,
        posemb_max_grid_size=posemb_max_grid_size,
    )

    transform = build_transform(cfg, input_format="pil")

    patch_dicts = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = _tensor_to_pil(img, normalise)
        patch_dicts.append(transform(img))

    batched_dict = patch_collate_fn([(d, 0) for d in patch_dicts])[0]
    batched_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batched_dict.items()}
    return batched_dict


def postprocess_images(
    output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_format: str = "minus_one_to_one",
    current_format: str = "minus_one_to_one",
    return_type: str = "tensor",
    unpack: bool = True,
    spatial_stride: int = 16,
    max_grid_size: Optional[int] = None,
) -> Union[torch.Tensor, List[Image.Image], List[np.ndarray]]:
    """Postprocess model output into user-friendly image formats.

    Args:
        output: Image tensor (B,C,H,W) or patch/output dict
        output_format: Target format ("minus_one_to_one", "zero_to_one", "0_255")
        current_format: Current format of the output
        return_type: Return type ("tensor", "pil", "numpy")
        unpack: Whether to unpack to original sizes
        spatial_stride: Patch size for unpatchify
        max_grid_size: Maximum grid size

    Returns:
        Processed images in requested format
    """
    if isinstance(output, dict):
        if 'images' in output:
            images = output['images']
        elif 'patches' in output:
            images = unpatchify(output, patch=spatial_stride, max_grid_size=max_grid_size)
        else:
            raise KeyError("postprocess_images expected 'images' or 'patches' in output dict")
    else:
        images = output

    if current_format is None:
        if images.dtype == torch.uint8:
            current_format = "0_255"
        elif images.min() >= -1.5 and images.max() <= 1.5:
            current_format = "minus_one_to_one"
        else:
            current_format = "zero_to_one"
    else:
        current_format = str(current_format)

    # Convert formats
    if output_format == "minus_one_to_one" and current_format != "minus_one_to_one":
        if current_format == "0_255":
            images = images.float() / 127.5 - 1.0
        elif current_format == "zero_to_one":
            images = images * 2.0 - 1.0
    elif output_format == "zero_to_one" and current_format != "zero_to_one":
        if current_format == "0_255":
            images = images.float() / 255.0
        elif current_format == "minus_one_to_one":
            images = (images + 1.0) / 2.0
    elif output_format == "0_255" and current_format != "0_255":
        if current_format == "minus_one_to_one":
            images = ((images + 1.0) / 2.0 * 255).round().to(torch.uint8)
        elif current_format == "zero_to_one":
            images = (images * 255).round().to(torch.uint8)

    if unpack:
        assert isinstance(output, dict), "postprocess_images(unpack=True) requires a patch/output dict"
        assert 'original_height' in output and 'original_width' in output
        unpack_dict = {
            'images': images,
            'original_height': output['original_height'],
            'original_width': output['original_width'],
        }
        return unpack_images(unpack_dict)
    else:
        return images


__all__ = [
    "NaFlexConfig",
    "build_naflex_transform",
    "naflex_batch",
    "preprocess_images",
    "postprocess_images",
]
