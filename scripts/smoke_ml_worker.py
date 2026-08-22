#!/usr/bin/env python3
"""Smoke the isolated ML worker without importing its packages in the caller."""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
from PIL import Image

from diffusion_editor.workers.ml_process import (
    MlProcessClient,
    default_worker_python,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--sdxl")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.real and not args.sdxl:
        parser.error("--real requires --sdxl")

    client = MlProcessClient(
        python=default_worker_python() if args.real else sys.executable,
        backend="real" if args.real else "fake",
        request_timeout=args.timeout,
    )
    cancel = threading.Event()
    requested_device = None if args.device == "auto" else args.device
    image = Image.new("RGB", (64, 64), "navy")
    try:
        client.request(
            "load_diffusion",
            {
                "model_path": args.sdxl or "fake.safetensors",
                "prediction_type": None,
                "device": requested_device,
            },
            cancel,
            on_progress=print,
        )
        diffusion = client.request(
            "diffusion",
            {
                "prompt": "a blue square",
                "negative_prompt": "",
                "strength": 1.0,
                "steps": 2,
                "guidance_scale": 1.0,
                "seed": 123,
                "mode": "img2img",
                "masked_content": "original",
                "ip_adapter_scale": 0.6,
                "width": 64,
                "height": 64,
            },
            cancel,
            images={"image": image, "mask": None, "ip_adapter": None},
            on_progress=print,
        )
        client.request(
            "load_instruct",
            {"device": requested_device},
            cancel,
            on_progress=print,
        )
        instruct = client.request(
            "instruct",
            {
                "instruction": "make it brighter",
                "guidance_scale": 7.5,
                "image_guidance_scale": 1.5,
                "steps": 1,
                "seed": 456,
            },
            cancel,
            images={"image": image},
            on_progress=print,
        )
        grounding = client.request(
            "grounding",
            {
                "prompt": "square",
                "model_id": "IDEA-Research/grounding-dino-tiny",
                "box_threshold": 0.1,
                "text_threshold": 0.1,
                "use_gpu": False,
                "sam2_model_id": None,
                "sam2_mask_channel": 0,
                "mask_threshold": 0.0,
                "max_hole_area": 0,
                "max_sprinkle_area": 0,
                "multimask": False,
                "non_overlap": False,
            },
            cancel,
            images={"image": np.asarray(image.convert("RGBA"))},
            on_progress=print,
        )
        depth = client.request(
            "depth",
            {
                "profile_id": "v2-small",
                "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                "backend": "transformers",
                "title": "Depth Anything V2 Small",
                "direct_depth": False,
                "value_kind": "inverse_relative",
            },
            cancel,
            images={"image": np.asarray(image.convert("RGBA"))},
            on_progress=print,
        )
        print(
            "ML worker smoke OK:",
            f"diffusion={diffusion['image'].size}",
            f"instruct={instruct['image'].size}",
            f"depth={depth['depth'].shape}/{depth['depth'].dtype}",
            f"detections={len(grounding['detections'])}",
        )
        return 0
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
