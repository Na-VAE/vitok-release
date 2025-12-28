"""Preprocessing module with string DSL for transform pipelines.

Example usage:
    from vitok.pp import build_transform, Registry

    # Build transform from string
    transform = build_transform(
        "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
    )
    patch_dict = transform(pil_image)

    # Register custom op
    @Registry.register("grayscale")
    def get_grayscale():
        return lambda img: img.convert("L").convert("RGB")

Available ops:
    - center_crop(size): Center crop to size x size
    - random_resized_crop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333)): Random resized crop
    - flip(p=0.5): Random horizontal flip
    - to_tensor: PIL to Tensor [0, 1]
    - normalize(mode): "minus_one_to_one" or "imagenet"
    - patchify(max_size, patch, max_tokens): Resize to budget + create patch dict
"""

from vitok.pp.registry import Registry, build_transform, parse_op

# Import ops module to register all ops
from vitok.pp import ops as _ops  # noqa: F401

__all__ = ["Registry", "build_transform", "parse_op"]
