"""HuggingFace streaming WebDataset wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from vitok.transforms import TransformCfg, build_transform, patch_collate_fn
from vitok.datasets.webdataset import HFWebDataset


@dataclass(frozen=True)
class StreamingWebDatasetConfig:
    """Configuration for HuggingFace streaming WebDataset.

    Args:
        hf_repo: HuggingFace dataset repository (e.g., "imagenet-1k")
        hf_revision: Repository revision/branch
        hf_subdir: Optional subdirectory in the repository
        hf_patterns: Optional glob patterns for shard files
        max_shards: Optional limit on number of shards
        batch_size: Batch size
        num_workers: Number of data loader workers
        prefetch_factor: Prefetch factor for data loader
        seed: Random seed
        train: Whether to apply training augmentations
        patch_size: Patch size in pixels
        max_tokens: Maximum sequence length
        train_max_grid_size: Max grid size for training
        posemb_max_grid_size: Max grid size for positional embeddings
        min_size: Minimum image dimension
        max_size: Maximum image dimension
        normalise: Normalization mode
        use_naflex_posemb: Enable NaFlex positional embeddings
        return_labels: Whether to return labels
        label_key: Key for label in WebDataset samples
    """

    hf_repo: str
    hf_revision: str = "main"
    hf_subdir: Optional[str] = None
    hf_patterns: Optional[Sequence[str]] = None
    max_shards: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 4
    seed: int = 0
    train: bool = True
    patch_size: int = 16
    max_tokens: int = 256
    train_max_grid_size: int = 48
    posemb_max_grid_size: int = 256
    min_size: int = 224
    max_size: int = 1024
    normalise: str = "minus_one_to_one"
    use_naflex_posemb: bool = True
    return_labels: bool = False
    label_key: str = "cls"


def create_streaming_dataloader(config: StreamingWebDatasetConfig):
    """Create a WebDataset dataloader backed by HuggingFace streaming shards.

    Args:
        config: StreamingWebDatasetConfig instance

    Returns:
        DataLoader that yields (patch_dict, labels) tuples
    """
    transform_cfg = TransformCfg(
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
    )
    transform = build_transform(transform_cfg, input_format="pil")

    dataset = HFWebDataset(
        hf_repo=config.hf_repo,
        hf_revision=config.hf_revision,
        hf_subdir=config.hf_subdir,
        hf_patterns=config.hf_patterns,
        max_shards=config.max_shards,
        transform=transform,
        collate_fn=patch_collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        seed=config.seed,
        min_size=config.min_size,
        patch_size=config.patch_size,
        max_tokens=config.max_tokens,
        max_grid_size=config.train_max_grid_size,
        return_labels=config.return_labels,
        label_key=config.label_key,
    )
    return dataset.create_dataloader()


__all__ = [
    "StreamingWebDatasetConfig",
    "create_streaming_dataloader",
]
