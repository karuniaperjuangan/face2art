"""Compact EnCo model and its original plus face-identity objectives.

Design adapted from Xiuding Cai et al.'s BSD-2-Clause EnCo implementation:
https://github.com/XiudingCai/EnCo-pytorch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .identity import FaceIdentityLoss
from .networks import PatchDiscriminator, ResnetGenerator, initialize_weights


class FeatureProjector(nn.Module):
    """Shared projection and decoder-side prediction heads for EnCo pairs."""

    def __init__(
        self,
        channels: list[int],
        projection_dim: int,
        predictor_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList()
        self.predictors = nn.ModuleList()
        for input_dim in channels:
            self.projectors.append(
                nn.Sequential(
                    nn.Linear(input_dim, projection_dim, bias=False),
                    nn.BatchNorm1d(projection_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(projection_dim, projection_dim, bias=False),
                    nn.BatchNorm1d(projection_dim, affine=False),
                )
            )
            self.predictors.append(
                nn.Sequential(
                    nn.Linear(projection_dim, predictor_hidden_dim, bias=False),
                    nn.BatchNorm1d(predictor_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(predictor_hidden_dim, projection_dim),
                )
            )


class DAGSampler:
    """Mix low-discriminator-response locations with uniform random locations."""

    def __init__(self, num_patches: int, oversample_ratio: int, random_ratio: float) -> None:
        if num_patches <= 0 or oversample_ratio <= 0 or not 0.0 <= random_ratio <= 1.0:
            raise ValueError("invalid DAG sampler configuration")
        self.num_patches = num_patches
        self.oversample_ratio = oversample_ratio
        self.random_ratio = random_ratio

    def choose(self, feature: Tensor, discriminator_map: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        scores = F.interpolate(
            discriminator_map, size=(height, width), mode="bilinear", align_corners=False
        ).mean(dim=1)
        count = min(self.num_patches, height * width)
        random_count = int(round(count * self.random_ratio))
        low_response_count = count - random_count
        result: list[Tensor] = []
        for batch_index in range(batch):
            candidate_count = min(height * width, count * self.oversample_ratio)
            candidates = torch.randperm(height * width, device=feature.device)[:candidate_count]
            ranked = torch.argsort(scores[batch_index].flatten()[candidates])
            low_response = candidates[ranked[:low_response_count]]
            uniform = torch.randperm(height * width, device=feature.device)[:random_count]
            result.append(torch.cat((low_response, uniform)))
        return torch.stack(result)

    @staticmethod
    def sample(feature: Tensor, indices: Tensor) -> Tensor:
        flattened = feature.flatten(2).transpose(1, 2)
        gathered = torch.gather(
            flattened,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, flattened.shape[-1]),
        )
        return gathered.flatten(0, 1)


@dataclass
class GeneratorOutput:
    translated: Tensor
    features: dict[str, Tensor]
    target_identity: Tensor | None


class EnCoModel(nn.Module):
    def __init__(
        self,
        generator: ResnetGenerator,
        discriminator: PatchDiscriminator,
        feature_network: FeatureProjector,
        feature_pairs: list[tuple[str, str]],
        sampler: DAGSampler,
        loss_config: dict[str, Any],
        identity_loss: FaceIdentityLoss | None = None,
        stop_gradient: bool = True,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_network = feature_network
        self.feature_pairs = feature_pairs
        self.sampler = sampler
        self.loss_config = loss_config
        self.identity_loss = identity_loss
        self.stop_gradient = stop_gradient
        self.lsgan = nn.MSELoss()

    def target_identity_active(self, epoch: int) -> bool:
        weight = float(self.loss_config["target_identity_weight"])
        return weight > 0 and epoch <= int(self.loss_config["target_identity_until_epoch"])

    def generate(self, source: Tensor, target: Tensor, epoch: int) -> GeneratorOutput:
        use_identity = self.target_identity_active(epoch)
        inputs = torch.cat((source, target), dim=0) if use_identity else source
        outputs, all_features = self.generator(inputs, return_features=True)
        batch = source.shape[0]
        features = {name: value[:batch] for name, value in all_features.items()}
        identity = outputs[batch:] if use_identity else None
        return GeneratorOutput(outputs[:batch], features, identity)

    def discriminator_loss(self, real_target: Tensor, fake_target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        prediction_real = self.discriminator(real_target)
        prediction_fake = self.discriminator(fake_target.detach())
        real = self.lsgan(prediction_real, torch.ones_like(prediction_real))
        fake = self.lsgan(prediction_fake, torch.zeros_like(prediction_fake))
        return 0.5 * (real + fake), {"d_real": real, "d_fake": fake}

    def content_loss(
        self, features: dict[str, Tensor], discriminator_map: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        losses: list[Tensor] = []
        patch_ids: list[Tensor] = []
        for index, (encoder_name, decoder_name) in enumerate(self.feature_pairs):
            encoder = features[encoder_name]
            decoder = features[decoder_name]
            indices = self.sampler.choose(decoder, discriminator_map)
            encoded_samples = self.sampler.sample(encoder, indices)
            decoded_samples = self.sampler.sample(decoder, indices)
            encoded_projection = self.feature_network.projectors[index](encoded_samples)
            decoded_projection = self.feature_network.projectors[index](decoded_samples)
            decoded_prediction = self.feature_network.predictors[index](decoded_projection)
            if self.stop_gradient:
                encoded_projection = encoded_projection.detach()
            loss = 2.0 - 2.0 * F.cosine_similarity(
                F.normalize(decoded_prediction, dim=1),
                F.normalize(encoded_projection, dim=1),
                dim=1,
            )
            losses.append(loss.mean())
            patch_ids.append(indices)
        return torch.stack(losses).mean(), patch_ids

    def generator_loss(
        self,
        source: Tensor,
        real_target: Tensor,
        output: GeneratorOutput,
    ) -> tuple[Tensor, dict[str, Tensor], list[Tensor]]:
        prediction_fake = self.discriminator(output.translated)
        gan = self.lsgan(prediction_fake, torch.ones_like(prediction_fake))
        content, patch_ids = self.content_loss(output.features, prediction_fake.detach())
        target_identity = (
            F.l1_loss(output.target_identity, real_target)
            if output.target_identity is not None
            else torch.zeros((), device=source.device)
        )
        if self.identity_loss is not None and float(self.loss_config["arcface_weight"]) > 0:
            face_identity = self.identity_loss(source, output.translated)
        else:
            face_identity = torch.zeros((), device=source.device)

        weighted_content = float(self.loss_config["enco_weight"]) * content
        weighted_target_identity = float(self.loss_config["target_identity_weight"]) * target_identity
        content_objective = (
            0.5 * (weighted_content + weighted_target_identity)
            if output.target_identity is not None
            else weighted_content
        )
        total = (
            float(self.loss_config["gan_weight"]) * gan
            + content_objective
            + float(self.loss_config["arcface_weight"]) * face_identity
        )
        losses = {
            "g_total": total,
            "g_gan": gan,
            "enco": content,
            "target_identity": target_identity,
            "arcface": face_identity,
        }
        return total, losses, patch_ids


def build_enco_model(
    config: dict[str, Any], identity_loss: FaceIdentityLoss | None = None
) -> EnCoModel:
    model = config["model"]
    generator = build_generator(model)
    if model["discriminator"] != "patchgan":
        raise ValueError(f"unsupported discriminator: {model['discriminator']}")
    discriminator = PatchDiscriminator(
        input_channels=int(model["output_channels"]),
        base_channels=int(model["discriminator_channels"]),
        layers=int(model["discriminator_layers"]),
        normalization=str(model["normalization"]),
    )
    pairs = [tuple(pair) for pair in model["feature_pairs"]]
    channels: list[int] = []
    for encoder_name, decoder_name in pairs:
        encoder_channels = generator.feature_channels[encoder_name]
        decoder_channels = generator.feature_channels[decoder_name]
        if encoder_channels != decoder_channels:
            raise ValueError(f"feature pair channels differ: {encoder_name}, {decoder_name}")
        channels.append(encoder_channels)
    feature_network = FeatureProjector(
        channels,
        projection_dim=int(model["projection_dim"]),
        predictor_hidden_dim=int(model["predictor_hidden_dim"]),
    )
    initialize_weights(discriminator, str(model["initialization"]), float(model["initialization_gain"]))
    initialize_weights(feature_network, str(model["initialization"]), float(model["initialization_gain"]))
    sampler = DAGSampler(
        int(model["num_patches"]),
        int(model["dag_oversample_ratio"]),
        float(model["dag_random_ratio"]),
    )
    return EnCoModel(
        generator,
        discriminator,
        feature_network,
        pairs,
        sampler,
        config["losses"],
        identity_loss=identity_loss,
        stop_gradient=bool(model["stop_gradient"]),
    )


def build_generator(model: dict[str, Any]) -> ResnetGenerator:
    if model["generator"] not in {"resnet_9blocks", "resnet_6blocks"}:
        raise ValueError(f"unsupported generator: {model['generator']}")
    blocks = 9 if model["generator"] == "resnet_9blocks" else 6
    generator = ResnetGenerator(
        input_channels=int(model["input_channels"]),
        output_channels=int(model["output_channels"]),
        base_channels=int(model["generator_channels"]),
        blocks=blocks,
        normalization=str(model["normalization"]),
        use_dropout=bool(model["use_dropout"]),
    )
    initialize_weights(generator, str(model["initialization"]), float(model["initialization_gain"]))
    return generator
