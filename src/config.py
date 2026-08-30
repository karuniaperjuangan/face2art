"""YAML loading, path resolution, validation, and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    required = {
        "experiment",
        "data",
        "model",
        "losses",
        "optimizer",
        "scheduler",
        "training",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"training config is missing sections: {', '.join(missing)}")

    data = config["data"]
    datasets = load_yaml(data["datasets_config"]).get("datasets", {})
    for role in ("source", "target"):
        name = data.get(role)
        if name not in datasets:
            raise ValueError(f"data.{role}={name!r} is not present in datasets config")
        if "path" not in datasets[name]:
            raise ValueError(f"dataset {name!r} has no path")

    if config["model"].get("name") != "enco":
        raise ValueError("this implementation currently supports model.name: enco")
    if config["losses"].get("gan_mode") != "lsgan":
        raise ValueError("this implementation currently supports losses.gan_mode: lsgan")
    if int(data.get("crop_size", 256)) % 4:
        raise ValueError("data.crop_size must be divisible by four")
    return config


def dataset_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    data = config["data"]
    datasets = load_yaml(data["datasets_config"])["datasets"]
    source = resolve_path(datasets[data["source"]]["path"])
    target = resolve_path(datasets[data["target"]]["path"])
    return source, target


def choose_device(value: str) -> torch.device:
    value = value.lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
