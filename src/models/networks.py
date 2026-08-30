"""Generator and discriminator building blocks used by EnCo.

The semantic feature names replace upstream EnCo's fragile numeric layer IDs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalization_factory(name: str) -> Callable[[int], nn.Module]:
    name = name.lower()
    if name == "instance":
        return lambda channels: nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
    if name == "batch":
        return lambda channels: nn.BatchNorm2d(channels)
    if name == "none":
        return lambda channels: nn.Identity()
    raise ValueError(f"unsupported normalization: {name}")


def initialize_weights(module: nn.Module, method: str = "normal", gain: float = 0.02) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if method == "normal":
                nn.init.normal_(layer.weight, 0.0, gain)
            elif method == "xavier":
                nn.init.xavier_normal_(layer.weight, gain=gain)
            elif method == "kaiming":
                nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
            else:
                raise ValueError(f"unsupported initialization: {method}")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if layer.affine:
                nn.init.normal_(layer.weight, 1.0, gain)
                nn.init.zeros_(layer.bias)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int, norm: Callable[[int], nn.Module], use_dropout: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            norm(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend(
            [
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3),
                norm(channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.block(inputs)


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, norm: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm = norm(output_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
        return self.activation(self.norm(self.conv(outputs)))


class ResnetGenerator(nn.Module):
    """CycleGAN-style ResNet generator with mirrored EnCo feature taps."""

    feature_names = (
        "enc_shallow",
        "enc_mid",
        "bottleneck_early",
        "bottleneck_late",
        "dec_mid",
        "dec_shallow",
    )

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        blocks: int = 9,
        normalization: str = "instance",
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if blocks < 4:
            raise ValueError("the EnCo generator needs at least four residual blocks")
        norm = normalization_factory(normalization)
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_channels, 7),
            norm(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            norm(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            norm(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            ResnetBlock(base_channels * 4, norm, use_dropout) for _ in range(blocks)
        )
        self.early_block = max(1, blocks // 3) - 1
        self.late_block = min(blocks - 1, blocks - max(1, blocks // 3))
        self.up1 = UpsampleBlock(base_channels * 4, base_channels * 2, norm)
        self.up2 = UpsampleBlock(base_channels * 2, base_channels, norm)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, output_channels, 7),
            nn.Tanh(),
        )
        self.feature_channels = {
            "enc_shallow": base_channels,
            "enc_mid": base_channels * 2,
            "bottleneck_early": base_channels * 4,
            "bottleneck_late": base_channels * 4,
            "dec_mid": base_channels * 2,
            "dec_shallow": base_channels,
        }

    def forward(
        self, inputs: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features: dict[str, Tensor] = {}
        outputs = self.stem(inputs)
        features["enc_shallow"] = outputs
        outputs = self.down1(outputs)
        features["enc_mid"] = outputs
        outputs = self.down2(outputs)
        for index, block in enumerate(self.blocks):
            outputs = block(outputs)
            if index == self.early_block:
                features["bottleneck_early"] = outputs
            if index == self.late_block:
                features["bottleneck_late"] = outputs
        outputs = self.up1(outputs)
        features["dec_mid"] = outputs
        outputs = self.up2(outputs)
        features["dec_shallow"] = outputs
        translated = self.head(outputs)
        if return_features:
            return translated, features
        return translated


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        layers: int = 3,
        normalization: str = "instance",
    ) -> None:
        super().__init__()
        norm = normalization_factory(normalization)
        sequence: list[nn.Module] = [
            nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        multiplier = 1
        for index in range(1, layers):
            previous = multiplier
            multiplier = min(2**index, 8)
            sequence.extend(
                [
                    nn.Conv2d(base_channels * previous, base_channels * multiplier, 4, stride=2, padding=1),
                    norm(base_channels * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        previous = multiplier
        multiplier = min(2**layers, 8)
        sequence.extend(
            [
                nn.Conv2d(base_channels * previous, base_channels * multiplier, 4, stride=1, padding=1),
                norm(base_channels * multiplier),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * multiplier, 1, 4, stride=1, padding=1),
            ]
        )
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)

