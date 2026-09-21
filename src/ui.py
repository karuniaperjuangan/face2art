"""Minimal Gradio UI for a trained face2art checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

from .config import PROJECT_ROOT, choose_device, load_training_config
from .infer import load_generator, translate_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/train.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def create_demo(config_path: Path, checkpoint_path: Path) -> gr.Interface:
    config = load_training_config(config_path)
    device = choose_device(str(config["training"]["device"]))
    generator = load_generator(config, checkpoint_path, device)
    size = int(config["data"]["crop_size"])

    def predict(image: Image.Image | None) -> Image.Image:
        if image is None:
            raise gr.Error("Upload a face image first.")
        return translate_image(image, generator, size, device)

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Face"),
        outputs=gr.Image(type="pil", label="Artwork"),
        title="Face2Art",
    )


def main() -> None:
    args = parse_args()
    create_demo(args.config, args.checkpoint).launch(share=args.share)


if __name__ == "__main__":
    main()
