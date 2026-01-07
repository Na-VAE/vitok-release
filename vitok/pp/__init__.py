"""Preprocessing module with string DSL for transform pipelines.

Example usage:
    from vitok.pp import build_transform, OPS

    # Build transform from string
    transform = build_transform(
        "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
    )
    patch_dict = transform(pil_image)

Available ops:
    - center_crop(size): Center crop to size x size
    - random_resized_crop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333)): Random resized crop
    - flip(p=0.5): Random horizontal flip
    - to_tensor: PIL to Tensor [0, 1]
    - normalize(mode): "minus_one_to_one" or "imagenet"
    - patchify(max_size, patch, max_tokens): Resize to budget + create patch dict
"""

from vitok.pp.ops import OPS, unpatchify, unpack, sample_tiles
from vitok.pp.registry import build_transform, parse_op
from vitok.pp.io import preprocess, postprocess

__all__ = ["build_transform", "parse_op", "OPS", "preprocess", "postprocess", "unpatchify", "unpack", "sample_tiles"]
