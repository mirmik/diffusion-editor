#!/usr/bin/env python3
"""Cache the spatial Qwen2.5-VL outputs used to condition Qwen Image Edit."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


DEFAULT_MODEL_SNAPSHOT = Path(
    "/home/mirmik/.cache/huggingface/hub/"
    "models--Qwen--Qwen-Image-Edit-2511/snapshots/"
    "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)
AZIMUTH_DESCRIPTORS = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}
ELEVATION_DESCRIPTORS = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot",
}
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_DROP_TOKENS = 64
CONDITION_SIZE = 384


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--view", action="append", dest="views")
    parser.add_argument("--layers", default="7,14,21,28")
    parser.add_argument(
        "--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    layers = [int(value.strip()) for value in args.layers.split(",")]
    if len(layers) != len(set(layers)) or any(value < 0 for value in layers):
        parser.error("--layers must contain unique non-negative indices")
    args.layers = layers
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _opaque_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if source.mode != "RGBA":
            return source.convert("RGB")
        background = Image.new("RGBA", source.size, (0, 0, 0, 255))
        background.alpha_composite(source)
        return background.convert("RGB")


def _anchor_view(views: list[dict], azimuth: float) -> dict:
    def distance(view: dict) -> tuple[float, float]:
        current = float(view["azimuth_degrees"]) % 360.0
        angular = abs((current - azimuth + 180.0) % 360.0 - 180.0)
        return angular, abs(float(view["elevation_degrees"]))

    return min(views, key=distance)


def _prompt(azimuth: float, elevation: float) -> str:
    rounded_azimuth = int(round(azimuth)) % 360
    rounded_elevation = int(round(elevation))
    azimuth_text = AZIMUTH_DESCRIPTORS.get(
        rounded_azimuth, f"azimuth {azimuth:g} degree view"
    )
    elevation_text = ELEVATION_DESCRIPTORS.get(
        rounded_elevation,
        (
            f"elevated {elevation:g} degree shot"
            if elevation > 0.0
            else f"low-angle shot {abs(elevation):g} degrees"
        ),
    )
    return f"<sks> {azimuth_text} {elevation_text} medium shot"


def _identity_id(dataset: dict, root: Path) -> str:
    asset = dataset.get("asset", {})
    return str(asset.get("identity_id") or root.name)


def main() -> int:
    args = _arguments()
    model_root = args.model_snapshot / "text_encoder"
    processor_root = args.model_snapshot / "processor"
    if not model_root.is_dir() or not processor_root.is_dir():
        raise SystemExit(f"incomplete model snapshot: {args.model_snapshot}")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-VL conditioning extraction requires CUDA")
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        processor_root, local_files_only=True
    )
    print("Loading complete Qwen2.5-VL text encoder...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    text_config = model.config.text_config
    hidden_size = int(text_config.hidden_size)
    layer_count = int(text_config.num_hidden_layers)
    if any(layer > layer_count for layer in args.layers):
        raise SystemExit(
            f"hidden-state layer exceeds available {layer_count}: {args.layers}"
        )
    image_token_id = int(model.config.image_token_id)
    merge = int(model.config.vision_config.spatial_merge_size)
    print(
        f"Loaded {layer_count} language blocks, hidden={hidden_size}, "
        f"VRAM={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )

    started = time.monotonic()
    completed_views = 0
    for dataset_path in args.datasets:
        dataset_root = dataset_path.resolve()
        dataset = _load_json(dataset_root / "manifest.json")
        identity = _identity_id(dataset, dataset_root)
        output_dir = args.output_root / identity
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists() and not args.force:
            raise SystemExit(f"output exists for {identity}: {manifest_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        views = list(dataset["views"])
        if args.views:
            requested = set(args.views)
            known = {view["id"] for view in views}
            missing = sorted(requested - known)
            if missing:
                raise SystemExit(f"unknown {identity} views: {missing}")
            views = [view for view in views if view["id"] in requested]

        front_view = _anchor_view(dataset["views"], 0.0)
        back_view = _anchor_view(dataset["views"], 180.0)
        source_images = []
        for view in (front_view, back_view):
            image = _opaque_rgb(dataset_root / view["rgb"])
            source_images.append(
                image.resize(
                    (CONDITION_SIZE, CONDITION_SIZE),
                    Image.Resampling.LANCZOS,
                )
            )

        output_views = []
        for view in views:
            prompt = _prompt(
                float(view["azimuth_degrees"]),
                float(view["elevation_degrees"]),
            )
            image_prompt = "".join(
                f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
                for index in (1, 2)
            )
            text = PROMPT_TEMPLATE.format(image_prompt + prompt)
            inputs = processor(
                text=[text],
                images=source_images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            grid_thw = inputs["image_grid_thw"]
            if tuple(grid_thw.shape) != (2, 3):
                raise RuntimeError(f"unexpected image grid: {grid_thw}")
            token_counts = [
                int(t * h * w) // (merge * merge)
                for t, h, w in grid_thw.tolist()
            ]
            merged_grids = [
                [int(w) // merge, int(h) // merge]
                for _t, h, w in grid_thw.tolist()
            ]
            if merged_grids[0] != merged_grids[1]:
                raise RuntimeError(f"source feature grids differ: {merged_grids}")
            print(
                f"Encoding {identity}/{view['id']} -> "
                f"2x{merged_grids[0][0]}x{merged_grids[0][1]}...",
                flush=True,
            )
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=grid_thw,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            valid = inputs["attention_mask"][0].bool()
            token_ids = inputs["input_ids"][0][valid][PROMPT_DROP_TOKENS:]
            image_positions = token_ids == image_token_id
            if int(image_positions.sum()) != sum(token_counts):
                raise RuntimeError(
                    f"image-token mismatch: found {int(image_positions.sum())}, "
                    f"expected {sum(token_counts)}"
                )
            layers = []
            for layer in args.layers:
                hidden = outputs.hidden_states[layer][0][valid][
                    PROMPT_DROP_TOKENS:
                ][image_positions]
                sources = torch.split(hidden, token_counts)
                layers.append(
                    torch.stack(
                        [
                            source.reshape(
                                merged_grids[index][1],
                                merged_grids[index][0],
                                hidden_size,
                            )
                            for index, source in enumerate(sources)
                        ]
                    )
                )
            conditioning = torch.stack(layers).float().cpu().numpy()
            relative = Path("views") / f"{view['id']}.npy"
            (output_dir / relative.parent).mkdir(parents=True, exist_ok=True)
            np.save(output_dir / relative, conditioning.astype(np.float16))
            output_views.append(
                {
                    "id": view["id"],
                    "azimuth_degrees": float(view["azimuth_degrees"]),
                    "elevation_degrees": float(view["elevation_degrees"]),
                    "prompt": prompt,
                    "conditioning": str(relative),
                    "shape": list(conditioning.shape),
                }
            )
            completed_views += 1
            del outputs, inputs, conditioning
            torch.cuda.empty_cache()

        manifest = {
            "schema": "diffusion-editor.qwen-vl-conditioning-features",
            "schema_version": 1,
            "dataset": str((dataset_root / "manifest.json").resolve()),
            "identity_id": identity,
            "model_snapshot": str(args.model_snapshot.resolve()),
            "source_views": [front_view["id"], back_view["id"]],
            "condition_size": [CONDITION_SIZE, CONDITION_SIZE],
            "prompt_template_drop_tokens": PROMPT_DROP_TOKENS,
            "hidden_state_layers": args.layers,
            "feature_channels": hidden_size,
            "source_feature_grid": merged_grids[0],
            "storage_dtype": "float16",
            "views": output_views,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved {identity}: {manifest_path}", flush=True)

    elapsed = time.monotonic() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"Complete: {completed_views} views in {elapsed:.1f}s "
        f"({elapsed / max(completed_views, 1):.2f}s/view)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
