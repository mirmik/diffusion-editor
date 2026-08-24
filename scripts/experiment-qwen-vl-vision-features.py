#!/usr/bin/env python3
"""Capture dense Qwen2.5-VL vision features from already rendered images."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MODEL_SNAPSHOT = Path(
    "/home/mirmik/.cache/huggingface/hub/"
    "models--Qwen--Qwen-Image-Edit-2511/snapshots/"
    "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)
GLOBAL_BLOCKS = (7, 15, 23, 31)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--labels", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize-square",
        type=int,
        default=0,
        help=(
            "Resize each raster to this square size before Qwen processing. "
            "Use 384 to mirror the square Image Edit conditioning path; zero "
            "keeps the source size."
        ),
    )
    return parser.parse_args()


def _scatter_raster(
    hidden: np.ndarray,
    positions: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    flat = positions[:, 0] * width + positions[:, 1]
    raster = np.empty((height * width, hidden.shape[-1]), dtype=np.float32)
    raster[flat] = hidden.astype(np.float32, copy=False)
    return raster.reshape(height, width, hidden.shape[-1])


def main() -> int:
    args = _arguments()
    if not args.model_snapshot.is_dir():
        raise SystemExit(f"missing model snapshot: {args.model_snapshot}")
    for path in args.images:
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    labels = (
        [value.strip() for value in args.labels.split(",")]
        if args.labels else [path.stem for path in args.images]
    )
    if len(labels) != len(args.images) or any(not value for value in labels):
        raise SystemExit("--labels must contain one non-empty label per image")
    if len(set(labels)) != len(labels):
        raise SystemExit("image labels must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    from transformers.vision_utils import (
        get_vision_position_ids,
        get_vision_window_index,
    )

    processor_path = args.model_snapshot / "processor"
    encoder_path = args.model_snapshot / "text_encoder"
    processor = AutoProcessor.from_pretrained(
        processor_path, local_files_only=True
    )
    config = Qwen2_5_VLConfig.from_pretrained(
        encoder_path, local_files_only=True
    ).vision_config

    print("Loading only the Qwen2.5-VL vision tower...", flush=True)
    visual = Qwen2_5_VisionTransformerPretrainedModel(config)
    index = json.loads(
        (encoder_path / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    visual_keys = {
        key: shard for key, shard in index.items()
        if key.startswith("visual.")
    }
    shards = sorted(set(visual_keys.values()))
    state = {}
    for shard in shards:
        with safe_open(
            encoder_path / shard, framework="pt", device="cpu"
        ) as archive:
            for key, key_shard in visual_keys.items():
                if key_shard == shard:
                    state[key.removeprefix("visual.")] = archive.get_tensor(key)
    incompatible = visual.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"vision state mismatch: {incompatible}")
    del state
    gc.collect()
    device = torch.device(args.device)
    visual.eval().to(device=device)
    feature_dtype = str(next(visual.parameters()).dtype).removeprefix("torch.")
    print(
        f"Vision tower: {len(visual.blocks)} blocks, "
        f"hidden={config.hidden_size}, weights={feature_dtype}",
        flush=True,
    )

    capture: dict[str, object] = {
        "enabled": False,
        "patch_store": None,
        "block_store": None,
        "q_store": None,
        "k_store": None,
        "original_positions": None,
        "internal_positions": None,
        "height": None,
        "width": None,
        "captured_blocks": set(),
        "captured_qk": set(),
    }
    global_slots = {block: slot for slot, block in enumerate(GLOBAL_BLOCKS)}

    def raster(tensor, *, internal: bool) -> np.ndarray:
        values = tensor.detach().float().cpu().numpy()
        positions = (
            capture["internal_positions"]
            if internal else capture["original_positions"]
        )
        return _scatter_raster(
            values,
            positions,
            int(capture["height"]),
            int(capture["width"]),
        )

    def patch_hook(_module, _inputs, output):
        if capture["enabled"]:
            capture["patch_store"][:] = raster(output, internal=False)

    def block_hook(block_index: int):
        def hook(_module, _inputs, output):
            if capture["enabled"]:
                capture["block_store"][block_index] = raster(
                    output, internal=True
                )
                capture["captured_blocks"].add(block_index)

        return hook

    def qkv_hook(block_index: int):
        def hook(_module, _inputs, output):
            if not capture["enabled"]:
                return
            query, key, _value = output.chunk(3, dim=-1)
            slot = global_slots[block_index]
            capture["q_store"][slot] = raster(query, internal=True)
            capture["k_store"][slot] = raster(key, internal=True)
            capture["captured_qk"].add(block_index)

        return hook

    handles = [visual.patch_embed.register_forward_hook(patch_hook)]
    for block_index, block in enumerate(visual.blocks):
        handles.append(block.register_forward_hook(block_hook(block_index)))
        if block_index in global_slots:
            handles.append(
                block.attn.qkv.register_forward_hook(qkv_hook(block_index))
            )

    manifest_views = []
    try:
        for image_path, label in zip(args.images, labels):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                original_size = [image.width, image.height]
                if args.resize_square:
                    image = image.resize(
                        (args.resize_square, args.resize_square),
                        Image.Resampling.LANCZOS,
                    )
                encoder_input_size = [image.width, image.height]
                inputs = processor(images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
            if grid_thw.shape != (1, 3) or int(grid_thw[0, 0]) != 1:
                raise RuntimeError(f"unexpected image grid: {grid_thw}")
            height, width = (int(value) for value in grid_thw[0, 1:])
            sequence = height * width
            original_positions = get_vision_position_ids(
                grid_thw, visual.spatial_merge_size
            )
            window_index, _window_cu = get_vision_window_index(
                grid_thw,
                spatial_merge_size=visual.spatial_merge_size,
                window_size=visual.window_size,
                patch_size=visual.patch_size,
            )
            merge_unit = visual.spatial_merge_unit
            internal_positions = (
                original_positions.reshape(-1, merge_unit, 2)[window_index]
                .reshape(sequence, 2)
            )
            capture["original_positions"] = original_positions.cpu().numpy()
            capture["internal_positions"] = internal_positions.cpu().numpy()
            capture["height"] = height
            capture["width"] = width
            capture["captured_blocks"] = set()
            capture["captured_qk"] = set()

            stem = label.replace("/", "-")
            patch_path = args.output_dir / f"patch-{stem}.npy"
            block_path = args.output_dir / f"blocks-{stem}.npy"
            q_path = args.output_dir / f"q-global-{stem}.npy"
            k_path = args.output_dir / f"k-global-{stem}.npy"
            capture["patch_store"] = np.lib.format.open_memmap(
                patch_path,
                mode="w+",
                dtype=np.float32,
                shape=(height, width, config.hidden_size),
            )
            capture["block_store"] = np.lib.format.open_memmap(
                block_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(visual.blocks), height, width, config.hidden_size),
            )
            capture["q_store"] = np.lib.format.open_memmap(
                q_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(GLOBAL_BLOCKS), height, width, config.hidden_size),
            )
            capture["k_store"] = np.lib.format.open_memmap(
                k_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(GLOBAL_BLOCKS), height, width, config.hidden_size),
            )

            print(
                f"Encoding {label}: source={original_size[0]}x{original_size[1]}, "
                f"grid={width}x{height}...",
                flush=True,
            )
            capture["enabled"] = True
            with torch.inference_mode():
                visual(
                    pixel_values.to(device=device, dtype=visual.dtype),
                    grid_thw=grid_thw.to(device),
                )
            capture["enabled"] = False
            expected_blocks = set(range(len(visual.blocks)))
            if capture["captured_blocks"] != expected_blocks:
                missing = sorted(expected_blocks - capture["captured_blocks"])
                raise RuntimeError(f"missing vision blocks: {missing}")
            if capture["captured_qk"] != set(GLOBAL_BLOCKS):
                missing = sorted(set(GLOBAL_BLOCKS) - capture["captured_qk"])
                raise RuntimeError(f"missing pre-RoPE Q/K blocks: {missing}")
            for key in ("patch_store", "block_store", "q_store", "k_store"):
                capture[key].flush()
                capture[key] = None
            manifest_views.append({
                "label": label,
                "image": str(image_path.resolve()),
                "original_size": original_size,
                "encoder_input_size": encoder_input_size,
                "processed_size": [width * config.patch_size,
                                   height * config.patch_size],
                "feature_grid": [width, height],
                "patch": str(patch_path.resolve()),
                "blocks": str(block_path.resolve()),
                "q_global": str(q_path.resolve()),
                "k_global": str(k_path.resolve()),
            })
            del pixel_values, grid_thw, inputs
            torch.cuda.empty_cache()
            print(f"Saved {label}", flush=True)
    finally:
        capture["enabled"] = False
        for handle in handles:
            handle.remove()

    manifest = {
        "model_snapshot": str(args.model_snapshot.resolve()),
        "model_component": "Qwen2.5-VL vision tower from text_encoder",
        "input_kind": "independent final rasters; no denoising/noise/seed/prompt",
        "resize_square": int(args.resize_square),
        "weight_dtype": feature_dtype,
        "storage_dtype": "float32",
        "patch_size": int(config.patch_size),
        "spatial_merge_size": int(config.spatial_merge_size),
        "feature_channels": int(config.hidden_size),
        "blocks": list(range(len(visual.blocks))),
        "global_attention_blocks": list(GLOBAL_BLOCKS),
        "views": manifest_views,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
