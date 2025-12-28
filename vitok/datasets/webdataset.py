"""WebDataset and HuggingFace streaming dataset support."""

import os
import random
from typing import Optional, Callable, List
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image, ImageOps

from vitok.transforms.image_transforms import fit_to_token_budget


def get_wds_tar_files_recursive(root_dirs):
    """Recursively find all *.tar files under the given root directories."""
    if isinstance(root_dirs, (str, Path)):
        root_dirs = [root_dirs]

    input_files = []
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if root_path.exists():
            tar_files = sorted(root_path.rglob("*.tar"))
            input_files.extend([str(f) for f in tar_files])
    return input_files


class WebDataset:
    """WebDataset wrapper for tar files containing images."""

    def __init__(
        self,
        bucket_paths: List[str],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_shuffle_buffer_size: int = 100000,
        seed: Optional[int] = None,
        prefetch_factor: int = 4,
        min_size: Optional[int] = None,
        patch_size: int = 16,
        max_tokens: int = 256,
        max_grid_size: Optional[int] = 48,
        return_labels: bool = False,
        label_key: str = "cls",
    ):
        if isinstance(bucket_paths, str):
            bucket_paths = [path.strip() for path in bucket_paths.split(',')]
        elif isinstance(bucket_paths, Path):
            bucket_paths = [str(bucket_paths)]

        self.bucket_paths = bucket_paths
        self.transform = transform
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_shuffle_buffer_size = sample_shuffle_buffer_size
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.min_size = min_size
        self.patch_size = patch_size
        self.max_tokens = max_tokens
        self.max_grid_size = max_grid_size
        self.return_labels = bool(return_labels)
        self.label_key = str(label_key)
        self.tar_files = self._collect_tar_files()

    def _collect_tar_files(self) -> List[str]:
        tar_files = get_wds_tar_files_recursive(self.bucket_paths)
        if not tar_files:
            raise ValueError(f"No tar files found in bucket paths: {self.bucket_paths}")
        return tar_files

    def _split_tar_files_by_rank(self, tar_files: List[str]) -> List[str]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return [tar_files[i] for i in range(rank, len(tar_files), world_size)]

    @staticmethod
    def _normalize_image_key(sample: dict) -> dict:
        for candidate in ("jpg", "jpeg", "png", "webp"):
            if candidate in sample:
                if candidate != "jpg":
                    sample["jpg"] = sample[candidate]
                return sample
        raise KeyError(f"No supported image key found in sample (keys: {list(sample.keys())})")

    @staticmethod
    def _has_supported_image_key(sample: dict) -> bool:
        for candidate in ("jpg", "jpeg", "png", "webp"):
            if candidate in sample:
                return True
        return False

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if img.mode == "P" and "transparency" in getattr(img, "info", {}):
            img = img.convert("RGBA")

        if img.mode in ("RGBA", "LA"):
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img

    def _resize_to_budget(self, sample):
        image = sample["jpg"]
        w, h = image.size
        new_h, new_w = fit_to_token_budget(h, w, self.patch_size, self.max_tokens, max_grid_size=self.max_grid_size)
        if (new_w, new_h) != (w, h):
            image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        sample["jpg"] = image
        return sample

    def _filter_by_size(self, sample):
        if self.min_size is None:
            return True
        image = sample["jpg"]
        w, h = image.size
        return min(w, h) >= self.min_size

    def _process_sample(self, sample):
        image = sample["jpg"]
        patch_dict = self.transform(image)

        label = 0
        if self.return_labels:
            raw = sample.get(self.label_key, None)
            if raw is not None:
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        label = int(raw.decode("utf-8").strip())
                    elif isinstance(raw, (str, int)):
                        label = int(raw)
                except Exception:
                    label = 0
        return patch_dict, label

    def create_dataloader(self) -> wds.WebLoader:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if self.seed is None:
            raise ValueError("seed must be provided for reproducible training")

        tar_rng = random.Random(self.seed)
        tar_files = self.tar_files[:]
        tar_rng.shuffle(tar_files)

        rank_tar_files = self._split_tar_files_by_rank(tar_files)
        sample_shuffle_seed = self.seed + rank

        dataset = (
            wds.WebDataset(
                rank_tar_files,
                handler=wds.ignore_and_continue,
                nodesplitter=wds.split_by_node,
                shardshuffle=False,
                resampled=True,
            )
            .shuffle(self.sample_shuffle_buffer_size, seed=sample_shuffle_seed)
            .decode("pil", handler=wds.ignore_and_continue)
            .select(WebDataset._has_supported_image_key)
            .map(WebDataset._normalize_image_key)
            .map_dict(jpg=WebDataset._ensure_rgb)
            .select(self._filter_by_size)
            .map(self._process_sample)
            .batched(self.batch_size, partial=False, collation_fn=self.collate_fn)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return loader

    def __len__(self):
        return len(self.tar_files)


class HFWebDataset(WebDataset):
    """WebDataset backed by HuggingFace Hub streaming shards.

    This class extends WebDataset to support streaming from HuggingFace Hub
    repositories containing tar shards.
    """

    def __init__(
        self,
        hf_repo: str,
        hf_revision: str = "main",
        hf_subdir: Optional[str] = None,
        hf_patterns: Optional[List[str]] = None,
        max_shards: Optional[int] = None,
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        seed: int = 0,
        min_size: int = 224,
        patch_size: int = 16,
        max_tokens: int = 256,
        max_grid_size: int = 48,
        return_labels: bool = False,
        label_key: str = "cls",
    ):
        self.hf_repo = hf_repo
        self.hf_revision = hf_revision
        self.hf_subdir = hf_subdir
        self.hf_patterns = hf_patterns or ["*.tar"]
        self.max_shards = max_shards

        # Get shard URLs from HuggingFace Hub
        shard_urls = self._get_hf_shard_urls()

        # Initialize parent with shard URLs as bucket_paths
        super().__init__(
            bucket_paths=[],  # Will be overridden
            transform=transform,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_shuffle_buffer_size=100000,
            seed=seed,
            prefetch_factor=prefetch_factor,
            min_size=min_size,
            patch_size=patch_size,
            max_tokens=max_tokens,
            max_grid_size=max_grid_size,
            return_labels=return_labels,
            label_key=label_key,
        )

        # Override tar_files with HF shard URLs
        self.tar_files = shard_urls

    def _get_hf_shard_urls(self) -> List[str]:
        """Get streaming URLs for tar shards from HuggingFace Hub."""
        try:
            from huggingface_hub import HfFileSystem
        except ImportError:
            raise ImportError("huggingface_hub is required for HFWebDataset")

        fs = HfFileSystem()

        # Build path prefix
        if self.hf_subdir:
            prefix = f"datasets/{self.hf_repo}/{self.hf_subdir}"
        else:
            prefix = f"datasets/{self.hf_repo}"

        # Find all matching tar files
        shard_urls = []
        for pattern in self.hf_patterns:
            glob_pattern = f"{prefix}/{pattern}"
            try:
                matches = fs.glob(glob_pattern)
                for match in matches:
                    # Convert to streaming URL
                    url = f"https://huggingface.co/{match.replace('datasets/', 'datasets/', 1)}/resolve/{self.hf_revision}/{match.split('/', 2)[-1]}"
                    # Simplified: use pipe: protocol for webdataset
                    url = f"pipe:curl -s -L {url}"
                    shard_urls.append(url)
            except Exception as e:
                print(f"Warning: Failed to glob {glob_pattern}: {e}")

        if self.max_shards:
            shard_urls = shard_urls[:self.max_shards]

        if not shard_urls:
            raise ValueError(f"No shards found for {self.hf_repo}")

        return shard_urls

    def _collect_tar_files(self) -> List[str]:
        # Override to skip local file collection
        return getattr(self, 'tar_files', [])
