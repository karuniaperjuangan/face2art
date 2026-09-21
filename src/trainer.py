"""A deliberately small EnCo training loop with logging and checkpoints."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

from .models.enco import EnCoModel


LOSS_COLUMNS = (
    "d_total",
    "d_real",
    "d_fake",
    "g_total",
    "g_gan",
    "enco",
    "target_identity",
    "arcface",
    "kid",
)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    max_arcface: float = float("inf")
    best_kid: float = float("inf")
    bad_checks: int = 0
    improved: bool = False

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early stopping patience must be positive")

    def step(self, kid: float, arcface: float) -> bool:
        self.improved = arcface <= self.max_arcface and kid < self.best_kid - self.min_delta
        if self.improved:
            self.best_kid = kid
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.bad_checks >= self.patience


def kernel_inception_distance(real: Tensor, fake: Tensor) -> float:
    """Unbiased polynomial-kernel MMD over Inception features."""
    if real.ndim != 2 or fake.ndim != 2 or min(len(real), len(fake)) < 2:
        raise ValueError("KID requires two 2D feature batches with at least two samples each")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake KID features must have the same width")
    real, fake = real.double(), fake.double()
    scale = real.shape[1]
    rr = (real @ real.T / scale + 1.0).pow(3)
    ff = (fake @ fake.T / scale + 1.0).pow(3)
    rf = (real @ fake.T / scale + 1.0).pow(3)
    real_term = (rr.sum() - rr.diagonal().sum()) / (len(real) * (len(real) - 1))
    fake_term = (ff.sum() - ff.diagonal().sum()) / (len(fake) * (len(fake) - 1))
    return float(real_term + fake_term - 2.0 * rf.mean())


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _scheduler_factor(epoch: int, warmup: int, constant: int, decay: int) -> float:
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / float(warmup)
    if epoch < constant:
        return 1.0
    if decay <= 0:
        return 0.0
    return max(0.0, 1.0 - float(epoch - constant + 1) / float(decay))


class Trainer:
    def __init__(
        self,
        model: EnCoModel,
        config: dict[str, Any],
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.sample_dir = output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "metrics.csv"
        self.early_stopping_log_path = output_dir / "early_stopping.csv"

        optimizer = config["optimizer"]
        kwargs = {
            "betas": (float(optimizer["beta1"]), float(optimizer["beta2"])),
            "weight_decay": float(optimizer["weight_decay"]),
        }
        self.optimizer_g = Adam(
            self.model.generator.parameters(), lr=float(optimizer["generator_lr"]), **kwargs
        )
        self.optimizer_f = Adam(
            self.model.feature_network.parameters(), lr=float(optimizer["feature_lr"]), **kwargs
        )
        self.optimizer_d = Adam(
            self.model.discriminator.parameters(), lr=float(optimizer["discriminator_lr"]), **kwargs
        )
        scheduler = config["scheduler"]
        factor = lambda epoch: _scheduler_factor(
            epoch,
            int(scheduler["warmup_epochs"]),
            int(scheduler["constant_epochs"]),
            int(scheduler["decay_epochs"]),
        )
        self.scheduler_g = LambdaLR(self.optimizer_g, factor)
        self.scheduler_f = LambdaLR(self.optimizer_f, factor)
        self.scheduler_d = LambdaLR(self.optimizer_d, factor)
        self.amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch = 1
        self.global_step = 0
        early = config.get("early_stopping", {})
        self.early_stopping = (
            EarlyStopping(
                patience=int(early["patience"]),
                min_delta=float(early.get("min_delta", 0.0)),
                max_arcface=float(early.get("max_arcface", float("inf"))),
            )
            if bool(early.get("enabled", False))
            else None
        )
        self._inception: torch.nn.Module | None = None
        self._real_kid_features: Tensor | None = None
        tracking = config.get("wandb", {})
        self.wandb_run: Any | None = None
        if bool(tracking.get("enabled", False)):
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("W&B logging requires the wandb package") from exc
            self.wandb_run = wandb.init(
                project=str(tracking.get("project", "face2art")),
                entity=tracking.get("entity") or None,
                name=tracking.get("name") or str(config["experiment"]["name"]),
                config=config,
                dir=str(output_dir),
            )

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp)

    def train_step(self, source: Tensor, target: Tensor, epoch: int) -> tuple[dict[str, float], Tensor]:
        source = source.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        self.model.train()
        if self.model.identity_loss is not None:
            self.model.identity_loss.eval()

        with self._autocast():
            output = self.model.generate(source, target, epoch)

        _set_requires_grad(self.model.discriminator, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        with self._autocast():
            d_total, d_parts = self.model.discriminator_loss(target, output.translated)
        self.scaler.scale(d_total).backward()
        self.scaler.step(self.optimizer_d)

        _set_requires_grad(self.model.discriminator, False)
        self.optimizer_g.zero_grad(set_to_none=True)
        self.optimizer_f.zero_grad(set_to_none=True)
        with self._autocast():
            g_total, g_parts, _ = self.model.generator_loss(source, target, output)
        self.scaler.scale(g_total).backward()
        gradient_clip = self.config["training"].get("gradient_clip_norm")
        if gradient_clip is not None:
            self.scaler.unscale_(self.optimizer_g)
            self.scaler.unscale_(self.optimizer_f)
            clip_grad_norm_(self.model.generator.parameters(), float(gradient_clip))
            clip_grad_norm_(self.model.feature_network.parameters(), float(gradient_clip))
        self.scaler.step(self.optimizer_g)
        self.scaler.step(self.optimizer_f)
        self.scaler.update()
        _set_requires_grad(self.model.discriminator, True)

        values = {
            "d_total": float(d_total.detach()),
            **{key: float(value.detach()) for key, value in d_parts.items()},
            **{key: float(value.detach()) for key, value in g_parts.items()},
        }
        return values, output.translated.detach()

    def _append_log(self, epoch: int, losses: dict[str, float]) -> None:
        new_file = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "step", *LOSS_COLUMNS))
            if new_file:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "step": self.global_step, **losses})

    def _log_wandb(self, prefix: str, epoch: int, metrics: dict[str, float]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    **{f"{prefix}/{key}": value for key, value in metrics.items()},
                }
            )

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _save_samples(self, source: Tensor, fake: Tensor, target: Tensor) -> None:
        batch = source.shape[0]
        grid = torch.cat(
            (source.detach().cpu(), fake.detach().cpu(), target.detach().cpu()), dim=0
        )
        save_image(
            grid,
            self.sample_dir / f"step_{self.global_step:08d}.png",
            nrow=batch,
            normalize=True,
            value_range=(-1, 1),
        )

    def _inception_features(self, images: Tensor) -> Tensor:
        if self._inception is None:
            weights = Inception_V3_Weights.DEFAULT
            self._inception = inception_v3(weights=weights)
            self._inception.fc = torch.nn.Identity()
            self._inception.eval().to(self.device)
        inputs = Inception_V3_Weights.DEFAULT.transforms()(
            images.float().add(1.0).div(2.0).clamp(0.0, 1.0)
        )
        return self._inception(inputs).flatten(1)

    @torch.inference_mode()
    def evaluate_early_stopping(self, loader: DataLoader) -> tuple[float, float]:
        if self.model.identity_loss is None:
            raise RuntimeError("ArcFace early stopping requires losses.arcface_weight > 0")
        self.model.generator.eval()
        self.model.identity_loss.eval()
        real_features: list[Tensor] = []
        fake_features: list[Tensor] = []
        identity_total = 0.0
        samples = 0
        for batch in loader:
            source = batch["source"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            fake = self.model.generator(source)
            if self._real_kid_features is None:
                real_features.append(self._inception_features(target).cpu())
            fake_features.append(self._inception_features(fake).cpu())
            identity_total += float(self.model.identity_loss(source, fake)) * len(source)
            samples += len(source)
        if samples < 2:
            raise ValueError("early stopping evaluation requires at least two samples")
        if self._real_kid_features is None:
            self._real_kid_features = torch.cat(real_features)
        kid = kernel_inception_distance(self._real_kid_features, torch.cat(fake_features))
        return kid, identity_total / samples

    def _append_early_stopping_log(self, epoch: int, kid: float, arcface: float) -> None:
        new_file = not self.early_stopping_log_path.exists()
        with self.early_stopping_log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(("epoch", "kid", "arcface"))
            writer.writerow((epoch, kid, arcface))

    def checkpoint(self, epoch: int, name: str = "latest.pt") -> Path:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": self.config,
                "generator": self.model.generator.state_dict(),
                "discriminator": self.model.discriminator.state_dict(),
                "feature_network": self.model.feature_network.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_f": self.optimizer_f.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "scheduler_g": self.scheduler_g.state_dict(),
                "scheduler_f": self.scheduler_f.state_dict(),
                "scheduler_d": self.scheduler_d.state_dict(),
                "scaler": self.scaler.state_dict(),
                "early_stopping": (
                    {
                        "best_kid": self.early_stopping.best_kid,
                        "bad_checks": self.early_stopping.bad_checks,
                    }
                    if self.early_stopping is not None
                    else None
                ),
            },
            path,
        )
        return path

    def resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.generator.load_state_dict(checkpoint["generator"])
        self.model.discriminator.load_state_dict(checkpoint["discriminator"])
        self.model.feature_network.load_state_dict(checkpoint["feature_network"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_f.load_state_dict(checkpoint["optimizer_f"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        self.scheduler_f.load_state_dict(checkpoint["scheduler_f"])
        self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        early = checkpoint.get("early_stopping")
        if early and self.early_stopping is not None:
            self.early_stopping.best_kid = float(early["best_kid"])
            self.early_stopping.bad_checks = int(early["bad_checks"])

    def fit(
        self,
        loader: DataLoader,
        max_steps: int | None = None,
        evaluation_loader: DataLoader | None = None,
    ) -> Path:
        training = self.config["training"]
        logging = self.config["logging"]
        checkpointing = self.config["checkpointing"]
        steps_this_run = 0
        last_epoch = self.start_epoch
        started = time.monotonic()
        for epoch in range(self.start_epoch, int(training["epochs"]) + 1):
            last_epoch = epoch
            for batch in loader:
                losses, fake = self.train_step(batch["source"], batch["target"], epoch)
                self.global_step += 1
                steps_this_run += 1
                self._append_log(epoch, losses)
                if self.global_step % int(logging["log_every_steps"]) == 0 or steps_this_run == 1:
                    elapsed = time.monotonic() - started
                    short = " ".join(
                        f"{key}={losses[key]:.4f}" for key in LOSS_COLUMNS if key in losses
                    )
                    print(f"epoch={epoch} step={self.global_step} time={elapsed:.1f}s {short}", flush=True)
                    self._log_wandb("train", epoch, losses)
                if self.global_step % int(logging["sample_every_steps"]) == 0 or steps_this_run == 1:
                    self._save_samples(batch["source"], fake, batch["target"])
                if max_steps is not None and steps_this_run >= max_steps:
                    return self.checkpoint(epoch)

            self.scheduler_g.step()
            self.scheduler_f.step()
            self.scheduler_d.step()
            early = self.config.get("early_stopping", {})
            should_check = (
                self.early_stopping is not None
                and epoch >= int(early.get("start_epoch", 1))
                and epoch % int(early.get("check_every_epochs", 1)) == 0
            )
            if should_check:
                if evaluation_loader is None:
                    raise ValueError("early stopping requires an evaluation loader")
                kid, arcface = self.evaluate_early_stopping(evaluation_loader)
                self._append_early_stopping_log(epoch, kid, arcface)
                self._append_log(epoch, {"kid": kid, "arcface": arcface})
                self._log_wandb("validation", epoch, {"kid": kid, "arcface": arcface})
                stop = self.early_stopping.step(kid, arcface)
                print(
                    f"epoch={epoch} validation kid={kid:.6f} arcface={arcface:.6f} "
                    f"patience={self.early_stopping.bad_checks}/{self.early_stopping.patience}",
                    flush=True,
                )
                if self.early_stopping.improved:
                    self.checkpoint(epoch, "best.pt")
                if stop:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    return self.checkpoint(epoch)
            if epoch % int(checkpointing["save_every_epochs"]) == 0:
                self.checkpoint(epoch, f"epoch_{epoch:04d}.pt")
            if bool(checkpointing["keep_latest"]):
                self.checkpoint(epoch)
        return self.checkpoint(last_epoch)
