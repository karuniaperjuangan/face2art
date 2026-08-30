"""Differentiable frozen AuraFace identity objective."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FaceIdentityLoss(nn.Module):
    """Cosine identity distance with gradients only through the generated image."""

    def __init__(self, backbone: nn.Module, input_size: int = 112) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_onnx(cls, path: Path, input_size: int = 112) -> "FaceIdentityLoss":
        try:
            from onnx2torch import convert
        except ImportError as exc:
            raise RuntimeError("onnx2torch is required for the AuraFace training loss") from exc
        return cls(convert(str(path)), input_size=input_size)

    def train(self, mode: bool = True) -> "FaceIdentityLoss":
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    @staticmethod
    def _unwrap(outputs: object) -> Tensor:
        if isinstance(outputs, Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], Tensor):
            return outputs[0]
        if isinstance(outputs, dict):
            first = next(iter(outputs.values()))
            if isinstance(first, Tensor):
                return first
        raise TypeError("AuraFace backbone returned an unsupported output")

    def embed(self, images: Tensor) -> Tensor:
        resized = F.interpolate(
            images.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        outputs = self._unwrap(self.backbone(resized))
        return F.normalize(outputs.flatten(1), dim=1)

    def forward(self, source: Tensor, generated: Tensor) -> Tensor:
        # The ONNX graph is kept in fp32; autograd still connects generated to G.
        context = (
            torch.autocast(device_type=generated.device.type, enabled=False)
            if generated.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            with torch.no_grad():
                source_embedding = self.embed(source)
            generated_embedding = self.embed(generated)
            return (1.0 - F.cosine_similarity(source_embedding, generated_embedding, dim=1)).mean()

