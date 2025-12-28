"""Local image dataset utilities."""

import os
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    """Simple dataset for loading images from a directory.

    Recursively finds all images in the root directory.
    Optionally assigns labels based on subdirectory names.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        with_labels: bool = False,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        """
        Args:
            root: Root directory containing images
            transform: Optional transform to apply to images
            with_labels: If True, assign labels based on folder names
            extensions: Tuple of valid image extensions
        """
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.with_labels = with_labels

        self.image_paths = sorted([
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(root)
            for f in fn if f.lower().endswith(self.extensions)
        ])

        if with_labels:
            self.samples = []
            self.class_to_idx = {}
            class_names = sorted({os.path.basename(os.path.dirname(p)) for p in self.image_paths})
            for idx, class_name in enumerate(class_names):
                self.class_to_idx[class_name] = idx
            for p in self.image_paths:
                class_name = os.path.basename(os.path.dirname(p))
                self.samples.append((p, self.class_to_idx[class_name]))
        else:
            self.samples = [(p, 0) for p in self.image_paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
