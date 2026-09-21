"""Flat-folder unpaired image dataset used by EnCo."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def discover_images(root: Path, extensions: Sequence[str], recursive: bool = False) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    suffixes = {f".{item.lower().lstrip('.')}" for item in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in suffixes)
    if not images:
        raise ValueError(f"no supported images found in {root}")
    return images


def make_train_transform(
    load_size: int,
    crop_size: int,
    random_crop: bool,
    horizontal_flip: bool,
) -> transforms.Compose:
    operations: list[object] = [
        transforms.Resize((load_size, load_size), InterpolationMode.BICUBIC, antialias=True)
    ]
    if random_crop:
        operations.append(transforms.RandomCrop(crop_size))
    else:
        operations.append(transforms.CenterCrop(crop_size))
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


class UnpairedImageDataset(Dataset[dict[str, Tensor | str]]):
    """Visit one domain sequentially and sample the other without identity pairing."""

    def __init__(
        self,
        source_root: Path,
        target_root: Path,
        extensions: Sequence[str],
        transform: transforms.Compose,
        epoch_size: str | int = "target",
        recursive: bool = False,
        random_source: bool = True,
    ) -> None:
        self.source_paths = discover_images(source_root, extensions, recursive)
        self.target_paths = discover_images(target_root, extensions, recursive)
        self.transform = transform
        self.random_source = random_source
        self.length = self._resolve_length(epoch_size)

    def _resolve_length(self, epoch_size: str | int) -> int:
        if isinstance(epoch_size, int):
            if epoch_size <= 0:
                raise ValueError("integer data.epoch_size must be positive")
            return epoch_size
        sizes = {"source": len(self.source_paths), "target": len(self.target_paths)}
        if epoch_size in sizes:
            return sizes[epoch_size]
        if epoch_size == "max":
            return max(sizes.values())
        if epoch_size == "min":
            return min(sizes.values())
        if str(epoch_size).isdigit() and int(epoch_size) > 0:
            return int(epoch_size)
        raise ValueError("data.epoch_size must be source, target, max, min, or a positive integer")

    @staticmethod
    def _open(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        target_path = self.target_paths[index % len(self.target_paths)]
        source_index = random.randrange(len(self.source_paths)) if self.random_source else index
        source_path = self.source_paths[source_index % len(self.source_paths)]
        source = self.transform(self._open(source_path))
        target = self.transform(self._open(target_path))
        return {
            "source": source,
            "target": target,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }

    def __len__(self) -> int:
        return self.length
