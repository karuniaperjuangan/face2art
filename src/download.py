#!/usr/bin/env python3
"""Download the datasets described by a YAML configuration.

Examples::

    python src/download.py --config config/datasets.yaml
    python src/download.py --dataset ffhq --limit 1000 --workers 16
    python src/download.py --asset auraface

FFHQ downloads use NVIDIA's public metadata and verify both the expected byte
count and MD5 checksum before atomically moving each image into ``data/``.
Google Drive folders are delegated to the optional ``gdown`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:  # Support both ``python src/download.py`` and ``python -m src.download``.
    from .assets import ensure_asset, load_assets
except ImportError:  # pragma: no cover - script entry point
    from assets import ensure_asset, load_assets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in a clean install
        raise SystemExit("PyYAML is required; install the project dependencies first") from exc
    return yaml


def load_config(path: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError(f"{path} must contain a mapping named 'datasets'")
    return config


def resolve_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def download_bytes(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = destination.with_name(f".{destination.name}.part")
        try:
            request = Request(url, headers={"User-Agent": "face2art-dataset-downloader/1.0"})
            with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drive_download_url(file_url: str) -> str:
    match = re.search(r"[?&]id=([^&]+)", file_url)
    if not match:
        return file_url
    return f"https://drive.usercontent.google.com/download?id={match.group(1)}&export=download&confirm=t"


def download_ffhq(spec: dict[str, Any], workers: int, limit: int | None, start: int | None) -> None:
    metadata_url = drive_download_url(str(spec["metadata_url"]))
    cache = resolve_path(str(spec.get("metadata_cache", "data/.cache/ffhq-dataset-v2.json")))
    expected_metadata_md5 = str(spec.get("metadata_md5", ""))
    if not cache.exists() or (expected_metadata_md5 and md5(cache) != expected_metadata_md5):
        print(f"Downloading FFHQ metadata to {cache}")
        download_bytes(metadata_url, cache)

    metadata = json.loads(cache.read_text(encoding="utf-8"))
    subset = spec.get("subset", {})
    first = int(start if start is not None else subset.get("start", 0))
    amount = int(limit if limit is not None else subset.get("count", len(metadata) - first))
    indices = range(first, min(first + amount, len(metadata)))
    output = resolve_path(str(spec["path"]))
    output.mkdir(parents=True, exist_ok=True)

    def fetch(index: int) -> tuple[int, str]:
        image = metadata[str(index)]["image"]
        target = output / f"{index:05d}.png"
        if target.is_file() and target.stat().st_size == image["file_size"] and md5(target) == image["file_md5"]:
            return index, "cached"
        download_bytes(drive_download_url(image["file_url"]), target)
        if target.stat().st_size != image["file_size"] or md5(target) != image["file_md5"]:
            target.unlink(missing_ok=True)
            raise IOError(f"checksum mismatch for FFHQ image {index:05d}")
        return index, "downloaded"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, index) for index in indices]
        for completed, future in enumerate(as_completed(futures), 1):
            index, status = future.result()
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"FFHQ {completed}/{len(futures)} ({index:05d}: {status})")


def download_google_drive_folder(spec: dict[str, Any], limit: int | None = None) -> None:
    output = resolve_path(str(spec["path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in a clean install
        raise SystemExit("The Google Drive dataset requires gdown (pip install gdown)") from exc
    if limit is not None:
        items = gdown.download_folder(url=str(spec["url"]), output=str(output), quiet=False, skip_download=True)
        for item in items[: max(0, limit)]:
            target = Path(item.local_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=item.id, output=str(target), quiet=False)
        return
    gdown.download_folder(url=str(spec["url"]), output=str(output), quiet=False)


def download_dataset(name: str, spec: dict[str, Any], args: argparse.Namespace) -> None:
    kind = str(spec.get("type", "")).lower()
    if kind == "ffhq":
        download_ffhq(spec, args.workers, args.limit, args.start)
    elif kind in {"google_drive_folder", "gdrive_folder"}:
        download_google_drive_folder(spec, args.limit)
    else:
        raise ValueError(f"Unsupported dataset type for {name!r}: {kind!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/datasets.yaml")
    parser.add_argument("--dataset", action="append", help="dataset name; repeat for multiple datasets (default: all)")
    parser.add_argument("--assets-config", type=Path, default=PROJECT_ROOT / "config/assets.yaml")
    parser.add_argument("--asset", action="append", help="model asset name; repeat for multiple assets")
    parser.add_argument("--force", action="store_true", help="redownload selected model assets")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="override the configured FFHQ subset count")
    parser.add_argument("--start", type=int, help="override the configured FFHQ subset start index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset or not args.asset:
        config = load_config(args.config.resolve())
        datasets = config["datasets"]
        selected = args.dataset or list(datasets)
        for name in selected:
            if name not in datasets:
                raise SystemExit(f"Unknown dataset {name!r}; choose from: {', '.join(datasets)}")
            print(f"==> {name}")
            download_dataset(name, datasets[name], args)

    if args.asset:
        assets = load_assets(args.assets_config)
        for name in args.asset:
            if name not in assets:
                raise SystemExit(f"Unknown asset {name!r}; choose from: {', '.join(assets)}")
            print(f"==> {name}")
            ensure_asset(name, assets[name], force=args.force)


if __name__ == "__main__":
    main()
