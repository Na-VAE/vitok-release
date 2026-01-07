"""Preprocessing module with string DSL for transform pipelines.

Example usage:
    from vitok.pp import build_transform, OPS

    # Build transform from string
    transform = build_transform(
        "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(256, 16)"
    )
    patch_dict = transform(pil_image)

    # Add custom op
    def grayscale():
        return lambda img: img.convert("L").convert("RGB")
    OPS["grayscale"] = grayscale

Available ops:
    - center_crop(size): Center crop to size x size
    - random_resized_crop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333)): Random resized crop
    - flip(p=0.5): Random horizontal flip
    - to_tensor: PIL to Tensor [0, 1]
    - normalize(mode): "minus_one_to_one" or "imagenet"
    - patchify(max_tokens, patch, max_grid_size, resize): Create patch dict
"""

from vitok.pp.registry import build_transform, parse_op
from vitok.pp.ops import OPS

__all__ = ["build_transform", "parse_op", "OPS"]
