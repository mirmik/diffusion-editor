"""Standalone loader for Comfy-style scaled FP8 safetensors checkpoints.

The checkpoint format is useful outside ComfyUI: model parameter names remain
the upstream names, FP8 linear weights carry a scalar dequantization scale, and
all precision-sensitive parameters stay in BF16/F32.  Keeping this adapter in
the worker avoids importing the ComfyUI runtime while reusing its model files.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any


KeyMapper = Callable[[str], str]


def identity_key(key: str) -> str:
    return key


class ScaledFP8LinearMixin:
    """Linear forward that stores a scaled FP8 weight and computes normally."""

    def _scaled_fp8_forward(self, input):
        import torch.nn.functional as functional

        weight = self.weight.to(dtype=input.dtype)
        weight.mul_(self._scaled_fp8_weight_scale.to(dtype=input.dtype))
        bias = self.bias
        if bias is not None and bias.dtype != input.dtype:
            bias = bias.to(dtype=input.dtype)
        return functional.linear(input, weight, bias)


def _scaled_linear_type():
    import torch

    class ScaledFP8Linear(ScaledFP8LinearMixin, torch.nn.Linear):
        def __init__(self, source: torch.nn.Linear) -> None:
            super().__init__(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
            self.weight.requires_grad_(False)
            if self.bias is not None:
                self.bias.requires_grad_(False)
            self.register_buffer(
                "_scaled_fp8_weight_scale",
                torch.empty((), device=source.weight.device, dtype=torch.float32),
                persistent=False,
            )

        def forward(self, input):
            return self._scaled_fp8_forward(input)

    return ScaledFP8Linear


def _quantized_layers(
    state: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
    """Return checkpoint layer name -> dequantization scale."""
    layers: dict[str, Any] = {}
    raw_metadata = metadata.get("_quantization_metadata")
    if raw_metadata:
        parsed = json.loads(raw_metadata)
        for name, config in parsed.get("layers", {}).items():
            if config.get("format") != "float8_e4m3fn":
                raise ValueError(
                    f"Unsupported scaled checkpoint format for {name}: "
                    f"{config.get('format')}"
                )
            scale = state.get(f"{name}.weight_scale")
            if scale is None:
                raise ValueError(f"Missing FP8 weight scale for layer: {name}")
            layers[name] = scale

    # Older scaled-FP8 text encoders use scale_weight/scale_input suffixes
    # instead of the JSON metadata block.
    for key, scale in state.items():
        if key.endswith(".scale_weight"):
            layers.setdefault(key.removesuffix(".scale_weight"), scale)
    return layers


def load_scaled_fp8_checkpoint(
    model,
    checkpoint: str | Path,
    *,
    key_mapper: KeyMapper = identity_key,
):
    """Load a mixed BF16/FP8 checkpoint into an empty (meta) model.

    Quantized Linear modules are replaced before ``assign=True`` loading so
    the source tensor dtypes are preserved rather than expanded to BF16.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(checkpoint).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"Scaled FP8 checkpoint not found: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    state = load_file(path, device="cpu")
    quantized = _quantized_layers(state, metadata)
    if not quantized:
        raise ValueError(f"No scaled FP8 layers found in checkpoint: {path}")

    scaled_linear = _scaled_linear_type()
    replacements: dict[str, Any] = {}
    for source_name, scale in quantized.items():
        name = key_mapper(source_name)
        try:
            source = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(
                f"FP8 checkpoint layer is absent from model: {source_name}"
            ) from exc
        if not isinstance(source, torch.nn.Linear):
            raise TypeError(
                f"FP8 checkpoint layer is not Linear: {source_name} "
                f"({type(source).__name__})"
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = scaled_linear(source)
        replacement.train(source.training)
        setattr(parent, child_name, replacement)
        replacements[name] = (replacement, scale)

    expected = set(model.state_dict())
    weights: dict[str, Any] = {}
    for source_key, tensor in state.items():
        key = key_mapper(source_key)
        if key in expected:
            weights[key] = tensor
    missing = expected - set(weights)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"Scaled FP8 checkpoint is missing {len(missing)} model weights: "
            f"{sample}"
        )

    incompatible = model.load_state_dict(weights, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Scaled FP8 checkpoint does not match the model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for name, (module, scale) in replacements.items():
        mapped_scale = scale.to(device=module.weight.device, dtype=torch.float32)
        module._buffers["_scaled_fp8_weight_scale"] = mapped_scale
        if module.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Scaled checkpoint weight is not FP8: {name} "
                f"({module.weight.dtype})"
            )
    model.eval()
    return model

