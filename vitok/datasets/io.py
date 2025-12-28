"""Image preprocessing and postprocessing utilities."""

import torch
import numpy as np
from typing import List, Union, Optional, Dict
from PIL import Image

from vitok.transforms import unpatchify, unpack_images, patch_collate_fn


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

    from vitok.transforms import TransformCfg, build_transform

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
            if img.ndim == 4:
                img = img.squeeze(0)
            if normalise == "minus_one_to_one":
                img = (img + 1.0) / 2.0
            img = torch.clamp(img, 0, 1)
            img = (img * 255).to(torch.uint8)
            img = Image.fromarray(img.cpu().numpy().transpose(1, 2, 0))

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
