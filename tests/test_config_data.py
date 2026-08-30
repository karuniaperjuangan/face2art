from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config import load_training_config
from src.data import UnpairedImageDataset, discover_images, make_train_transform


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color).save(path)


def test_project_training_config_resolves_dataset_names() -> None:
    config = load_training_config("config/train.yaml")
    assert config["data"]["source"] == "ffhq"
    assert config["data"]["target"] == "sngfaces"
    assert config["model"]["name"] == "enco"


def test_flat_unpaired_dataset_and_normalization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _image(source / "a.png", (255, 0, 0))
    _image(source / "b.jpg", (0, 255, 0))
    _image(target / "painting.png", (0, 0, 255))
    (source / "ignored.txt").write_text("not an image", encoding="utf-8")

    assert len(discover_images(source, ["png", "jpg"])) == 2
    transform = make_train_transform(20, 16, random_crop=False, horizontal_flip=False)
    dataset = UnpairedImageDataset(source, target, ["png", "jpg"], transform, epoch_size="target")
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["source"].shape == (3, 16, 16)
    assert sample["target"].shape == (3, 16, 16)
    assert torch.all(sample["source"].abs() <= 1.0)
    assert torch.all(sample["target"].abs() <= 1.0)


def test_dataset_rejects_invalid_epoch_size(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _image(source / "a.png", (1, 2, 3))
    _image(target / "b.png", (4, 5, 6))
    transform = make_train_transform(16, 16, False, False)
    with pytest.raises(ValueError, match="epoch_size"):
        UnpairedImageDataset(source, target, ["png"], transform, epoch_size="unknown")

