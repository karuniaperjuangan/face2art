from __future__ import annotations

import torch
from PIL import Image

from src.infer import translate_image


def test_translate_image_returns_pil_image() -> None:
    image = Image.new("RGB", (12, 10), (255, 0, 0))
    result = translate_image(image, torch.nn.Identity(), 8, torch.device("cpu"))
    assert result.mode == "RGB"
    assert result.size == (8, 8)
