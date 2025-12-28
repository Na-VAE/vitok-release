"""Image transforms with NaFlex token budget support."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List
import math
import random

from vitok.transforms.patch_ops import create_patch_dict
from vitok.transforms.positional_encoding import PositionalEncoding2D


class CreatePatchDict:
    """Picklable wrapper for create_patch_dict function."""

    def __init__(self, patch_size, max_seq_len, posemb_encoder):
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.posemb_encoder = posemb_encoder

    def __call__(self, x):
        return create_patch_dict(x, self.patch_size, max_seq_len=self.max_seq_len, posemb_encoder=self.posemb_encoder)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style centre crop with box downsampling."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ADMCenterCrop:
    """Picklable wrapper for ADM centre crop."""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        return center_crop_arr(pil_image, self.image_size)


def fit_to_token_budget(
    h: int,
    w: int,
    patch: int,
    max_tokens: int,
    max_grid_size: int | None = None,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    """Find the largest (h', w') <= (h, w) that fits within token budget."""
    h_p = math.ceil(h / patch)
    w_p = math.ceil(w / patch)

    within_tokens = (h_p * w_p) <= max_tokens
    within_grid = (max_grid_size is None) or (h_p <= max_grid_size and w_p <= max_grid_size)
    if within_tokens and within_grid:
        return h, w

    scale = math.sqrt(max_tokens / (h_p * w_p))
    new_h_p = max(1, math.floor(h_p * scale + eps))
    new_w_p = max(1, math.floor(w_p * scale + eps))

    new_h = new_h_p * patch
    new_w = new_w_p * patch

    if max_grid_size is not None:
        if new_h_p > max_grid_size or new_w_p > max_grid_size:
            scale = min(max_grid_size / new_h_p, max_grid_size / new_w_p)
            new_h_p = max(1, math.floor(new_h_p * scale + eps))
            new_w_p = max(1, math.floor(new_w_p * scale + eps))
            new_h = new_h_p * patch
            new_w = new_w_p * patch

    return min(new_h, h), min(new_w, w)


@dataclass
class TransformCfg:
    """Configuration for image transforms."""

    train: bool = True
    patch_size: int = 16
    min_size: int = 224
    max_size: int = 512
    center_crop_prob: float = 0.1
    normalise: str = "minus_one_to_one"
    max_tokens: int = 256
    train_max_grid_size: int = 64
    posemb_max_grid_size: int = 256
    temperature: float = 10000.0
    augmentation_strategy: str = "naflex"
    use_naflex_posemb: bool = False
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    target_token_distribution: str = "uniform"
    native_image_prob: float = 0.5
    square_crop_prob: float = 0.1
    square_crop_sizes: Tuple[int, ...] = (256, 512)
    true_random_crop_prob: float = 0.5


class RandomResizeCropWithBudget:
    """Randomly resize-crop using min/max bounds, then enforce token budget."""

    def __init__(self, cfg: TransformCfg):
        self.cfg = cfg

    def __call__(self, img):
        patch = self.cfg.patch_size
        width, height = img.size

        if random.random() < self.cfg.square_crop_prob:
            square_size = random.choice(self.cfg.square_crop_sizes)
            if min(width, height) >= square_size:
                max_x = width - square_size
                max_y = height - square_size
                x = random.randint(0, max(0, max_x))
                y = random.randint(0, max(0, max_y))
                img = TF.crop(img, y, x, square_size, square_size)
            else:
                img = TF.resize(img, [square_size, square_size],
                               interpolation=TF.InterpolationMode.LANCZOS, antialias=True)

        elif random.random() < self.cfg.true_random_crop_prob:
            min_crop = self.cfg.min_size
            max_crop_h = min(height, self.cfg.train_max_grid_size * patch)
            max_crop_w = min(width, self.cfg.train_max_grid_size * patch)

            if max_crop_h >= min_crop and max_crop_w >= min_crop:
                rmin, rmax = self.cfg.aspect_ratio_range
                crop_h = random.randint(min_crop, max_crop_h)
                min_w = max(min_crop, int(crop_h * rmin))
                max_w = min(max_crop_w, int(crop_h * rmax))
                crop_w = random.randint(min_w, max_w) if min_w <= max_w else min_w

                crop_h_p = math.ceil(crop_h / patch)
                crop_w_p = math.ceil(crop_w / patch)
                if crop_h_p * crop_w_p > self.cfg.max_tokens:
                    scale = math.sqrt(self.cfg.max_tokens / (crop_h_p * crop_w_p))
                    crop_h = int(crop_h * scale)
                    crop_w = int(crop_w * scale)

                max_x = width - crop_w
                max_y = height - crop_h
                if max_x >= 0 and max_y >= 0:
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    img = TF.crop(img, y, x, crop_h, crop_w)

        elif random.random() < self.cfg.center_crop_prob:
            crop_size = random.randint(self.cfg.min_size, self.cfg.max_size)
            img = TF.center_crop(img, crop_size)
        else:
            min_grid = max(1, math.ceil(self.cfg.min_size / patch))
            max_grid = max(min_grid, math.ceil(self.cfg.max_size / patch))
            grid_h = random.randint(min_grid, max_grid)
            grid_w = random.randint(min_grid, max_grid)
            target_h = int(grid_h * patch)
            target_w = int(grid_w * patch)

            rmin, rmax = self.cfg.aspect_ratio_range
            crop = T.RandomResizedCrop(
                size=(target_h, target_w),
                scale=self.cfg.crop_scale_range,
                ratio=(rmin, rmax),
                interpolation=TF.InterpolationMode.LANCZOS,
                antialias=True,
            )
            img = crop(img)

        width, height = img.size
        new_h, new_w = fit_to_token_budget(
            height, width, patch, self.cfg.max_tokens,
            max_grid_size=self.cfg.train_max_grid_size,
        )
        if (new_h, new_w) != (height, width):
            img = TF.resize(img, [new_h, new_w],
                           interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
        return img


class ResizeLongestSideWithBudget:
    """Resize image to fixed long side before enforcing token budget."""

    def __init__(self, cfg: TransformCfg):
        self.cfg = cfg

    def __call__(self, img):
        target = int(self.cfg.max_size)
        width, height = img.size
        long_side = max(width, height)
        if long_side != target:
            scale = target / float(long_side)
            new_h = max(1, int(round(height * scale)))
            new_w = max(1, int(round(width * scale)))
            img = TF.resize(img, [new_h, new_w],
                           interpolation=TF.InterpolationMode.LANCZOS, antialias=True)

        width, height = img.size
        patch = self.cfg.patch_size
        new_h, new_w = fit_to_token_budget(
            height, width, patch, self.cfg.max_tokens,
            max_grid_size=self.cfg.train_max_grid_size,
        )
        if (new_h, new_w) != (height, width):
            img = TF.resize(img, [new_h, new_w],
                           interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
        return img


def build_transform(cfg: TransformCfg, input_format: str = "pil") -> T.Compose:
    """Build transform pipeline.

    Args:
        cfg: Transform configuration
        input_format: "pil" for PIL images

    Returns:
        Composed transform pipeline
    """
    trs: List = []

    if cfg.train:
        if cfg.augmentation_strategy in ("square", "naflex_square"):
            trs.append(ADMCenterCrop(cfg.max_size))
        elif cfg.augmentation_strategy == "naflex":
            trs.append(RandomResizeCropWithBudget(cfg))
        else:
            raise ValueError(f"Unknown augmentation strategy: {cfg.augmentation_strategy}")

        trs.append(T.RandomHorizontalFlip())
        trs.append(T.ToTensor())
    else:
        if cfg.augmentation_strategy in ("square", "naflex_square"):
            trs.append(ADMCenterCrop(cfg.max_size))
        elif cfg.augmentation_strategy == "naflex":
            trs.append(ResizeLongestSideWithBudget(cfg))
        else:
            raise ValueError(f"Unknown augmentation strategy: {cfg.augmentation_strategy}")
        trs.append(T.ToTensor())

    if cfg.normalise == "minus_one_to_one":
        trs.append(T.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
    elif cfg.normalise == "imagenet":
        trs.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if cfg.augmentation_strategy != "square":
        max_seq_len = cfg.max_tokens
        if cfg.use_naflex_posemb:
            embed_dim = cfg.patch_size * cfg.patch_size * 3
            posemb_encoder = PositionalEncoding2D(
                embed_dim=embed_dim,
                max_grid_size=cfg.posemb_max_grid_size,
                temperature=cfg.temperature,
                dtype=torch.float32,
                max_seq_len=max_seq_len
            )
        else:
            posemb_encoder = None

        trs.append(CreatePatchDict(cfg.patch_size, max_seq_len, posemb_encoder))

    return T.Compose(trs)


def unpack_images(patches_dict: dict) -> list:
    """Unpack images from patch dict to their original sizes."""
    images = patches_dict['images']
    orig_h = patches_dict['original_height']
    orig_w = patches_dict['original_width']
    if images.ndim == 3:
        images = images.unsqueeze(0)

    cropped = []
    for img, h, w in zip(images, orig_h, orig_w):
        h_val = int(h.item() if isinstance(h, torch.Tensor) else h)
        w_val = int(w.item() if isinstance(w, torch.Tensor) else w)
        cropped_img = img[:, :h_val, :w_val]
        cropped.append(cropped_img)
    return cropped
