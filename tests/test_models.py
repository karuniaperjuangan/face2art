from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from src.config import load_training_config
from src.models import build_enco_model
from src.trainer import EarlyStopping, Trainer, kernel_inception_distance


def tiny_config() -> dict:
    config = deepcopy(load_training_config("config/train.yaml"))
    config["model"].update(
        {
            "generator": "resnet_6blocks",
            "generator_channels": 8,
            "discriminator_channels": 8,
            "projection_dim": 16,
            "predictor_hidden_dim": 8,
            "num_patches": 4,
        }
    )
    config["losses"]["arcface_weight"] = 0.0
    config["training"]["mixed_precision"] = False
    config["training"]["epochs"] = 2
    config["logging"]["log_every_steps"] = 1
    config["logging"]["sample_every_steps"] = 1
    return config


def test_generator_mirrored_feature_shapes() -> None:
    model = build_enco_model(tiny_config())
    translated, features = model.generator(torch.randn(1, 3, 32, 32), return_features=True)
    assert translated.shape == (1, 3, 32, 32)
    for encoder, decoder in model.feature_pairs:
        assert features[encoder].shape == features[decoder].shape


def test_enco_losses_are_finite_and_backpropagate() -> None:
    model = build_enco_model(tiny_config())
    source = torch.randn(1, 3, 32, 32)
    target = torch.randn(1, 3, 32, 32)
    output = model.generate(source, target, epoch=1)
    d_total, d_parts = model.discriminator_loss(target, output.translated)
    g_total, g_parts, patch_ids = model.generator_loss(source, target, output)
    assert torch.isfinite(d_total)
    assert all(torch.isfinite(value) for value in d_parts.values())
    assert all(torch.isfinite(value) for value in g_parts.values())
    assert len(patch_ids) == 3
    assert all(indices.shape == (1, 4) for indices in patch_ids)
    g_total.backward()
    assert any(parameter.grad is not None for parameter in model.generator.parameters())


def test_trainer_checkpoint_round_trip(tmp_path: Path) -> None:
    config = tiny_config()
    trainer = Trainer(build_enco_model(config), config, torch.device("cpu"), tmp_path / "run")
    losses, fake = trainer.train_step(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32), 1)
    trainer.global_step = 1
    checkpoint = trainer.checkpoint(1)
    assert fake.shape == (1, 3, 32, 32)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())

    restored = Trainer(build_enco_model(config), config, torch.device("cpu"), tmp_path / "restored")
    restored.resume(checkpoint)
    assert restored.global_step == 1
    assert restored.start_epoch == 2
    first = next(trainer.model.generator.parameters()).detach()
    second = next(restored.model.generator.parameters()).detach()
    assert torch.equal(first, second)


def test_early_stopping_uses_kid_with_arcface_guardrail() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.01, max_arcface=0.5)
    assert not stopping.step(kid=0.2, arcface=0.4)
    assert stopping.improved
    assert not stopping.step(kid=0.1, arcface=0.6)
    assert stopping.step(kid=0.195, arcface=0.4)


def test_kernel_inception_distance_detects_a_shift() -> None:
    real = torch.randn(16, 8)
    assert kernel_inception_distance(real, real + 5) > kernel_inception_distance(real, real)
