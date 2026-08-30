"""Translate one image or a flat image folder with a trained EnCo generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image

from .config import PROJECT_ROOT, choose_device, load_training_config, resolve_path
from .data import discover_images
from .models import build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/train.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    device = choose_device(str(config["training"]["device"]))
    generator = build_generator(config["model"]).to(device)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    input_path = resolve_path(args.input)
    if input_path.is_dir():
        inputs = discover_images(
            input_path, config["data"]["extensions"], bool(config["data"]["recursive"])
        )
    elif input_path.is_file():
        inputs = [input_path]
    else:
        raise FileNotFoundError(input_path)

    output_path = resolve_path(args.output)
    if len(inputs) > 1 or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        destinations = [output_path / path.name for path in inputs]
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        destinations = [output_path]

    size = int(config["data"]["crop_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((size, size), InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    with torch.inference_mode():
        for source, destination in zip(inputs, destinations, strict=True):
            with Image.open(source) as image:
                inputs_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            translated = generator(inputs_tensor)[0].detach().cpu().add(1).div(2).clamp(0, 1)
            to_pil_image(translated).save(destination)
            print(destination)


if __name__ == "__main__":
    main()

