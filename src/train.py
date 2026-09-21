"""Train EnCo on unpaired FFHQ and SNGFaces folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .assets import ensure_asset, load_assets, resolve_path as resolve_asset_path
from .config import (
    PROJECT_ROOT,
    choose_device,
    dataset_paths,
    load_training_config,
    resolve_path,
    seed_everything,
    seed_worker,
)
from .data import UnpairedImageDataset, make_train_transform
from .models import FaceIdentityLoss, build_enco_model
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/train.yaml")
    parser.add_argument("--max-steps", type=int, help="stop after this many steps in this run")
    parser.add_argument("--resume", type=Path, help="checkpoint override")
    return parser.parse_args()


def build_loader(config: dict, device: torch.device, *, training: bool = True) -> DataLoader:
    data = config["data"]
    early = config.get("early_stopping", {})
    source_root, target_root = dataset_paths(config)
    transform = make_train_transform(
        int(data["load_size"]),
        int(data["crop_size"]),
        bool(data["random_crop"]) if training else False,
        bool(data["horizontal_flip"]) if training else False,
    )
    dataset = UnpairedImageDataset(
        source_root,
        target_root,
        data["extensions"],
        transform,
        epoch_size=data["epoch_size"] if training else int(early["samples"]),
        recursive=bool(data["recursive"]),
        random_source=training,
    )
    workers = int(data["num_workers"])
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(data["batch_size"] if training else early.get("batch_size", 8)),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(data["pin_memory"]) and device.type == "cuda",
        persistent_workers=bool(data["persistent_workers"]) and workers > 0,
        drop_last=bool(data["drop_last"]) if training else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def load_identity_loss(config: dict) -> FaceIdentityLoss | None:
    if float(config["losses"]["arcface_weight"]) <= 0:
        return None
    arcface = config["arcface"]
    assets = load_assets(arcface["assets_config"])
    name = str(arcface["asset"])
    spec = assets[name]
    path = resolve_asset_path(spec["path"])
    if not path.is_file():
        if not bool(arcface["auto_download"]):
            raise FileNotFoundError(f"missing {name}; run python src/download.py --asset {name}")
        path = ensure_asset(name, spec)
    print(f"Converting frozen AuraFace graph: {path}", flush=True)
    return FaceIdentityLoss.from_onnx(path, input_size=int(arcface["input_size"]))


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = choose_device(str(config["training"]["device"]))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"Using {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"Using {device}", flush=True)

    identity = load_identity_loss(config)
    model = build_enco_model(config, identity_loss=identity)
    loader = build_loader(config, device)
    evaluation_loader = (
        build_loader(config, device, training=False)
        if bool(config.get("early_stopping", {}).get("enabled", False))
        else None
    )
    output_dir = resolve_path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    trainer = Trainer(model, config, device, output_dir)
    try:
        resume = args.resume or config["training"].get("resume")
        if resume:
            trainer.resume(resolve_path(resume))
        checkpoint = trainer.fit(
            loader, max_steps=args.max_steps, evaluation_loader=evaluation_loader
        )
    finally:
        trainer.close()
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
