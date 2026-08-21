from __future__ import annotations

import json

import pytest

from diffusion_editor.workers.scaled_fp8 import _quantized_layers


def test_quantized_layers_reads_current_metadata_format():
    scale = object()
    metadata = {
        "_quantization_metadata": json.dumps({
            "format_version": "1.0",
            "layers": {
                "block.proj": {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": True,
                },
            },
        }),
    }

    assert _quantized_layers(
        {"block.proj.weight_scale": scale}, metadata
    ) == {"block.proj": scale}


def test_quantized_layers_reads_legacy_scaled_text_encoder_format():
    scale = object()
    assert _quantized_layers(
        {"model.layers.0.mlp.up_proj.scale_weight": scale}, {}
    ) == {"model.layers.0.mlp.up_proj": scale}


def test_quantized_layers_rejects_unknown_format():
    metadata = {
        "_quantization_metadata": json.dumps({
            "layers": {"block.proj": {"format": "nvfp4"}},
        }),
    }
    with pytest.raises(ValueError, match="Unsupported scaled checkpoint"):
        _quantized_layers({}, metadata)
