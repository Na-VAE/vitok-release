"""Collate functions for patch-based batching."""

import torch


def patch_collate_fn(batch):
    """Collate function for patchified data.

    Handles:
    - (dict, label) tuples from patchified pipelines
    - (tensor, label) tuples from square image pipelines
    - plain dicts or tensors (no labels)
    """
    if len(batch) == 0:
        return {}, None

    if isinstance(batch[0], tuple):
        patch_dicts, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long) if labels[0] is not None else None
    else:
        patch_dicts = batch
        labels = None

    if isinstance(patch_dicts[0], torch.Tensor):
        return torch.stack(patch_dicts, dim=0), labels

    collated = {}
    for k in patch_dicts[0].keys():
        items = [d[k] for d in patch_dicts]
        if isinstance(items[0], torch.Tensor):
            collated[k] = torch.stack(items, dim=0)
        else:
            collated[k] = items

    return collated, labels
