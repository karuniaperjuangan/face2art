from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.assets import sha256
from src.models import FaceIdentityLoss


def test_identity_loss_only_backpropagates_to_generated_image() -> None:
    backbone = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 8),
    )
    loss_module = FaceIdentityLoss(backbone, input_size=16)
    source = torch.randn(2, 3, 16, 16, requires_grad=True)
    generated = torch.randn(2, 3, 16, 16, requires_grad=True)
    loss = loss_module(source, generated)
    loss.backward()
    assert source.grad is None
    assert generated.grad is not None
    assert generated.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


@pytest.mark.integration
def test_auraface_conversion_matches_onnx_and_has_input_gradients() -> None:
    value = os.environ.get("AURAFACE_MODEL")
    if not value:
        pytest.skip("set AURAFACE_MODEL to run the 261 MB integration test")
    path = Path(value)
    assert sha256(path) == "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"

    ort = pytest.importorskip("onnxruntime")
    identity = FaceIdentityLoss.from_onnx(path)
    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        converted = identity._unwrap(identity.backbone(inputs)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})[0]
    np.testing.assert_allclose(converted, expected, rtol=2e-4, atol=2e-4)

    generated = torch.randn(1, 3, 112, 112, requires_grad=True)
    identity(inputs, generated).backward()
    assert generated.grad is not None
    assert generated.grad.abs().sum() > 0

