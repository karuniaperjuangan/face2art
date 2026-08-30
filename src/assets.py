"""Download and validate external model assets declared in YAML."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assets(path: str | Path) -> dict[str, dict[str, Any]]:
    with resolve_path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    assets = config.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"{path} must contain an assets mapping")
    return assets


def ensure_asset(name: str, spec: dict[str, Any], force: bool = False) -> Path:
    destination = resolve_path(spec["path"])
    expected = str(spec.get("sha256", "")).lower()
    if destination.is_file() and not force:
        if not expected or sha256(destination) == expected:
            print(f"Using cached asset: {destination}")
            return destination
        raise IOError(f"cached asset has the wrong SHA-256: {destination}")

    if spec.get("type") != "huggingface":
        raise ValueError(f"unsupported asset type for {name!r}: {spec.get('type')!r}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download model assets") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from {spec['repo_id']} at {spec['revision']}")
    downloaded = Path(
        hf_hub_download(
            repo_id=str(spec["repo_id"]),
            filename=str(spec["filename"]),
            revision=str(spec["revision"]),
            local_dir=destination.parent,
            force_download=force,
        )
    )
    if downloaded.resolve() != destination.resolve():
        raise RuntimeError(f"asset downloaded to unexpected path: {downloaded}")
    actual = sha256(destination)
    if expected and actual != expected:
        destination.unlink(missing_ok=True)
        raise IOError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    print(f"Verified {name}: {actual}")
    return destination
