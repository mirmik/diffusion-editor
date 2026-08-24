"""Small trainable head used by the canonical point-map experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = math.gcd(channels, 16)
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values):
        return values + self.layers(values)


class CanonicalPointmapHead(nn.Module):
    def __init__(
        self,
        blocks: int,
        feature_channels: int,
        projection_channels: int,
        hidden: int,
        ray_channels: int,
        timestep_channels: int = 0,
    ):
        super().__init__()
        self.ray_channels = ray_channels
        self.timestep_channels = timestep_channels
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_channels, projection_channels, 1),
                    nn.SiLU(),
                )
                for _ in range(blocks)
            ]
        )
        self.camera = nn.Sequential(
            nn.Linear(16, hidden),
            nn.SiLU(),
            nn.Linear(hidden, projection_channels),
        )
        if timestep_channels:
            self.register_buffer(
                "timestep_frequencies",
                2.0 ** torch.arange(8, dtype=torch.float32),
            )
            self.timestep_embedding = nn.Sequential(
                nn.Linear(17, hidden),
                nn.SiLU(),
                nn.Linear(hidden, timestep_channels),
            )
        else:
            self.timestep_frequencies = None
            self.timestep_embedding = None
        fused = (
            blocks * projection_channels
            + projection_channels
            + ray_channels
            + timestep_channels
        )
        self.input = nn.Conv2d(fused, hidden, 3, padding=1)
        self.low_resolution = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        middle = max(hidden // 2, 32)
        self.up = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, middle, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(middle, 5, 1)

    def forward(
        self,
        features,
        camera,
        rays,
        output_size: int,
        timestep=None,
    ):
        projected = [
            layer(features[:, index])
            for index, layer in enumerate(self.projectors)
        ]
        camera_features = self.camera(camera)[:, :, None, None]
        camera_features = camera_features.expand(
            -1, -1, features.shape[-2], features.shape[-1]
        )
        inputs = [*projected, camera_features]
        if self.ray_channels:
            inputs.append(rays)
        if self.timestep_channels:
            if timestep is None:
                raise ValueError("a timestep-conditioned head requires timestep input")
            values = timestep.reshape(-1, 1).to(
                device=features.device,
                dtype=features.dtype,
            )
            angles = math.pi * values * self.timestep_frequencies[None]
            encoded = torch.cat((values, torch.sin(angles), torch.cos(angles)), dim=1)
            timestep_features = self.timestep_embedding(encoded)[:, :, None, None]
            inputs.append(
                timestep_features.expand(
                    -1, -1, features.shape[-2], features.shape[-1]
                )
            )
        values = self.low_resolution(self.input(torch.cat(inputs, dim=1)))
        values = functional.interpolate(
            values,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.output(self.up(values))


class ConditionedResidualBlock(nn.Module):
    """Residual convolution with camera/timestep FiLM conditioning."""

    def __init__(self, channels: int, condition_channels: int):
        super().__init__()
        groups = math.gcd(channels, 16)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_channels, channels * 2)

    def forward(self, values, condition):
        scale, shift = self.condition(condition).chunk(2, dim=1)
        normalized = self.norm1(values)
        normalized = normalized * (1.0 + scale[:, :, None, None])
        normalized = normalized + shift[:, :, None, None]
        result = self.conv1(functional.silu(normalized))
        result = self.conv2(functional.silu(self.norm2(result)))
        return values + result


class ConditionedStage(nn.Module):
    def __init__(self, channels: int, condition_channels: int, blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConditionedResidualBlock(channels, condition_channels)
                for _ in range(blocks)
            ]
        )

    def forward(self, values, condition):
        for block in self.blocks:
            values = block(values, condition)
        return values


class SpatialAttentionBlock(nn.Module):
    """Global attention at the compact 16x16 bottleneck."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, values):
        batch, channels, height, width = values.shape
        tokens = values.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(
            self.feed_forward_norm(tokens)
        )
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class VLContextAttentionBlock(nn.Module):
    """Let compact target tokens query spatial front/back Qwen2.5-VL tokens."""

    def __init__(self, channels: int, layers: int, context_channels: int):
        super().__init__()
        self.layers = layers
        self.context_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(context_channels),
                    nn.Linear(context_channels, channels),
                )
                for _index in range(layers)
            ]
        )
        self.layer_embedding = nn.Parameter(torch.zeros(layers, channels))
        self.position = nn.Sequential(
            nn.Linear(4, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            8,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        # A zero gate makes a v2 backbone upgrade prediction-preserving.  The
        # first optimizer step learns whether context is useful; subsequent
        # steps then train the projections and attention through the open gate.
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _positions(self, context):
        _batch, _layers, sources, height, width, _channels = context.shape
        rows, columns = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=context.device),
            torch.linspace(-1.0, 1.0, width, device=context.device),
            indexing="ij",
        )
        positions = []
        for source in range(sources):
            source_front = torch.full_like(rows, float(source == 0))
            source_back = torch.full_like(rows, float(source == 1))
            positions.append(
                torch.stack((columns, rows, source_front, source_back), dim=-1)
            )
        return self.position(torch.stack(positions))

    def forward(self, values, context):
        if context is None:
            raise ValueError("a VL-fusion head requires Qwen2.5-VL context")
        if context.ndim != 6 or context.shape[1] != self.layers:
            raise ValueError(
                "VL context must have shape "
                f"(batch, {self.layers}, sources, height, width, channels)"
            )
        batch, channels, height, width = values.shape
        position = self._positions(context).to(dtype=context.dtype)
        projected = []
        for index, projector in enumerate(self.context_projectors):
            tokens = projector(context[:, index])
            tokens = tokens + position[None]
            tokens = tokens + self.layer_embedding[index][None, None, None, None]
            projected.append(tokens.flatten(1, 3))
        context_tokens = torch.cat(projected, dim=1)
        target_tokens = values.flatten(2).transpose(1, 2)
        attended, _weights = self.attention(
            self.query_norm(target_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            need_weights=False,
        )
        contextual = target_tokens + attended
        delta = attended + self.feed_forward(
            self.feed_forward_norm(contextual)
        )
        target_tokens = target_tokens + self.gate * delta
        return target_tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )


class PredictionBranches(nn.Module):
    """Keep geometry, silhouette and uncertainty refinements independent."""

    def __init__(self, channels: int):
        super().__init__()

        def branch(outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, outputs, 1),
            )

        self.xyz = branch(3)
        self.mask = branch(1)
        self.uncertainty = branch(1)

    def forward(self, values):
        return torch.cat(
            (self.xyz(values), self.mask(values), self.uncertainty(values)),
            dim=1,
        )


class MultiscaleCanonicalPointmapHead(nn.Module):
    """Multiscale 64->16->256 canonical point-map decoder."""

    def __init__(
        self,
        blocks: int,
        feature_channels: int,
        projection_channels: int,
        hidden: int,
        ray_channels: int,
        timestep_channels: int = 0,
        context_layers: int = 0,
        context_channels: int = 0,
    ):
        super().__init__()
        self.ray_channels = ray_channels
        self.timestep_channels = timestep_channels
        self.context_layers = context_layers
        self.context_channels = context_channels
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_channels, projection_channels, 1),
                    nn.SiLU(),
                )
                for _ in range(blocks)
            ]
        )

        condition_channels = hidden * 2
        self.camera_condition = nn.Sequential(
            nn.Linear(16, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
        )
        if timestep_channels:
            self.register_buffer(
                "timestep_frequencies",
                2.0 ** torch.arange(8, dtype=torch.float32),
            )
            self.timestep_condition = nn.Sequential(
                nn.Linear(17, condition_channels),
                nn.SiLU(),
                nn.Linear(condition_channels, condition_channels),
            )
        else:
            self.timestep_frequencies = None
            self.timestep_condition = None

        level32 = hidden + hidden // 2
        level16 = hidden * 2
        fused_channels = blocks * projection_channels + ray_channels
        self.stem = nn.Conv2d(fused_channels, hidden, 3, padding=1)
        self.encoder64 = ConditionedStage(hidden, condition_channels, blocks=2)
        self.down32 = nn.Conv2d(hidden, level32, 3, stride=2, padding=1)
        self.encoder32 = ConditionedStage(level32, condition_channels, blocks=2)
        self.down16 = nn.Conv2d(level32, level16, 3, stride=2, padding=1)
        self.bottleneck = ConditionedStage(level16, condition_channels, blocks=3)
        self.attention = SpatialAttentionBlock(level16)
        self.vl_context = (
            VLContextAttentionBlock(
                level16,
                context_layers,
                context_channels,
            )
            if context_layers and context_channels
            else None
        )

        self.up32 = nn.Conv2d(level16, level32, 3, padding=1)
        self.merge32 = nn.Conv2d(level32 * 2, level32, 3, padding=1)
        self.decoder32 = ConditionedStage(level32, condition_channels, blocks=1)
        self.up64 = nn.Conv2d(level32, hidden, 3, padding=1)
        self.merge64 = nn.Conv2d(hidden * 2, hidden, 3, padding=1)
        self.decoder64 = ConditionedStage(hidden, condition_channels, blocks=1)

        level128 = max(hidden * 3 // 4, 64)
        level256 = max(hidden // 2, 48)
        self.refine128 = nn.Sequential(
            nn.ConvTranspose2d(hidden, level128, 4, stride=2, padding=1),
            nn.GroupNorm(math.gcd(level128, 16), level128),
            nn.SiLU(),
        )
        self.condition128 = ConditionedStage(
            level128, condition_channels, blocks=1
        )
        self.refine256 = nn.Sequential(
            nn.ConvTranspose2d(level128, level256, 4, stride=2, padding=1),
            nn.GroupNorm(math.gcd(level256, 16), level256),
            nn.SiLU(),
        )
        self.condition256 = ConditionedStage(
            level256, condition_channels, blocks=1
        )
        self.auxiliary64 = PredictionBranches(hidden)
        self.auxiliary128 = PredictionBranches(level128)
        self.output = PredictionBranches(level256)

    def _condition(self, camera, timestep, dtype):
        condition = self.camera_condition(camera.to(dtype=dtype))
        if self.timestep_channels:
            if timestep is None:
                raise ValueError(
                    "a timestep-conditioned head requires timestep input"
                )
            values = timestep.reshape(-1, 1).to(
                device=camera.device,
                dtype=dtype,
            )
            angles = math.pi * values * self.timestep_frequencies[None]
            encoded = torch.cat(
                (values, torch.sin(angles), torch.cos(angles)), dim=1
            )
            condition = condition + self.timestep_condition(encoded)
        return condition

    def forward_pyramid(
        self,
        features,
        camera,
        rays,
        output_size: int,
        timestep=None,
        context=None,
    ):
        projected = [
            layer(features[:, index])
            for index, layer in enumerate(self.projectors)
        ]
        inputs = projected
        if self.ray_channels:
            inputs = [*inputs, rays]
        condition = self._condition(camera, timestep, features.dtype)

        level64 = self.encoder64(self.stem(torch.cat(inputs, dim=1)), condition)
        level32 = self.encoder32(self.down32(level64), condition)
        level16 = self.bottleneck(self.down16(level32), condition)
        level16 = self.attention(level16)
        if self.vl_context is not None:
            level16 = self.vl_context(level16, context)

        decoded32 = functional.interpolate(
            level16,
            size=level32.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded32 = self.up32(decoded32)
        decoded32 = self.decoder32(
            self.merge32(torch.cat((decoded32, level32), dim=1)),
            condition,
        )
        decoded64 = functional.interpolate(
            decoded32,
            size=level64.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded64 = self.up64(decoded64)
        decoded64 = self.decoder64(
            self.merge64(torch.cat((decoded64, level64), dim=1)),
            condition,
        )
        decoded128 = self.condition128(self.refine128(decoded64), condition)
        decoded256 = self.condition256(self.refine256(decoded128), condition)

        prediction64 = self.auxiliary64(decoded64)
        prediction128 = self.auxiliary128(decoded128)
        prediction = self.output(decoded256)
        if prediction.shape[-2:] != (output_size, output_size):
            prediction = functional.interpolate(
                prediction,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "prediction": prediction,
            "auxiliary": (prediction64, prediction128),
        }

    def forward(
        self,
        features,
        camera,
        rays,
        output_size: int,
        timestep=None,
        context=None,
    ):
        return self.forward_pyramid(
            features,
            camera,
            rays,
            output_size,
            timestep=timestep,
            context=context,
        )["prediction"]


def build_canonical_pointmap_head(
    architecture: str,
    blocks: int,
    feature_channels: int,
    projection_channels: int,
    hidden: int,
    ray_channels: int,
    timestep_channels: int = 0,
    context_layers: int = 0,
    context_channels: int = 0,
):
    if architecture == "local-v1":
        return CanonicalPointmapHead(
            blocks,
            feature_channels,
            projection_channels,
            hidden,
            ray_channels,
            timestep_channels,
        )
    if architecture == "multiscale-v2":
        return MultiscaleCanonicalPointmapHead(
            blocks,
            feature_channels,
            projection_channels,
            hidden,
            ray_channels,
            timestep_channels,
        )
    if architecture == "multiscale-vl-v3":
        if not context_layers or not context_channels:
            raise ValueError("multiscale-vl-v3 requires spatial VL context")
        return MultiscaleCanonicalPointmapHead(
            blocks,
            feature_channels,
            projection_channels,
            hidden,
            ray_channels,
            timestep_channels,
            context_layers,
            context_channels,
        )
    raise ValueError(f"unknown canonical point-map head architecture: {architecture}")
