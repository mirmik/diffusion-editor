#!/usr/bin/env python3
"""Exercise real Qwen 2.1 through the ML backend; write RGBA PNGs and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from diffusion_editor.generation.text_to_image_profiles import text_to_image_profile
from diffusion_editor.workers.ml_backend import RealMlBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/diffusion-editor-qwen21-smoke"))
    parser.add_argument("--steps", type=int, default=4)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--turbo8", action="store_true")
    profiles.add_argument("--turbo", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeat-size", type=int, default=0,
                        help="generate again at this size without reloading")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    profile_id = ("qwen-image-2.1-turbo" if args.turbo else
                  "qwen-image-2.1-turbo8" if args.turbo8 else "qwen-image-2.1")
    profile = text_to_image_profile(profile_id)
    p = profile.defaults()
    if args.turbo8 or args.turbo:
        args.steps = 8
    if args.check_only:
        from diffusion_editor.workers.qwen_image21_process import QwenImage21Pipeline
        p["device"] = "cpu"
        pipe = QwenImage21Pipeline(p, check_only=True)
        try:
            print(json.dumps(pipe.runtime_info, indent=2))
        finally:
            pipe.close()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    p.update(steps=args.steps, seed=42,
             prompt="This is an RGBA image with transparency. A red ceramic teapot. The image has alpha channel and the background is transparent.")
    backend = RealMlBackend()
    data = {"profile_id": profile_id, "parameters": p,
            "lora_adapters": [a.to_dict() for a in profile.default_lora_adapters]}
    started = time.monotonic()
    try:
        print("Loading Qwen Image 2.1", flush=True)
        info = backend.load_text_to_image(data)
        print(json.dumps(info, indent=2), flush=True)
        result, seed, provenance = backend.text_to_image(dict(data, width=args.size, height=args.size))
        assert result.size == (args.size, args.size) and result.mode == "RGBA"
        result.save(args.output / "text-to-image.png")
        if args.steps >= 30 or args.turbo8 or args.turbo:
            assert result.getextrema()[3][0] < 32, "No transparent background generated"
        (args.output / "text-to-image.json").write_text(json.dumps(provenance, indent=2))
        print("Text-to-image:", result.size, result.getextrema(), flush=True)
        if args.repeat_size:
            size = args.repeat_size
            result, seed, provenance = backend.text_to_image(dict(data, width=size, height=size))
            assert result.size == (size, size) and result.mode == "RGBA"
            if args.turbo8:
                sampling = provenance["runtime"]["provider_runtime"]["sampling"]
                expected_shift = 0.5 + (size * size / 256 - 256) * (0.6935 - 0.5) / 3840
                assert abs(sampling["shift"] - expected_shift) < 1e-8
                assert sampling["steps"] == 8 and sampling["width"] == size
            result.save(args.output / "text-to-image-repeat.png")
            (args.output / "text-to-image-repeat.json").write_text(json.dumps(provenance, indent=2))
            print("Repeated generation:", result.size, result.getextrema(), flush=True)
        # Reference-only generation: different subject, non-square reference,
        # no source pixels from the target layer, exact non-multiple output size.
        reference = Image.new("RGBA", (768, 384), (245, 226, 185, 255))
        drawing = ImageDraw.Draw(reference)
        drawing.rectangle((0, 256, 768, 384), fill=(20, 85, 91, 255))
        drawing.ellipse((170, 36, 430, 296), fill=(221, 79, 36, 255), outline=(24, 39, 39, 255), width=12)
        drawing.line((80, 310, 650, 80), fill=(24, 39, 39, 255), width=15)
        reference.save(args.output / "style-reference.png")
        for reference_seed in (42, 43):
            ref_parameters = dict(p, seed=reference_seed,
                prompt="Create a new illustration of a lighthouse on a rocky coast with ocean waves. Use <image1> only as an artistic style reference: flat geometric shapes, thick dark outlines, muted cream, burnt orange and deep teal palette. Create a new composition. Do not reproduce the circle or diagonal line from the reference.")
            result, seed, provenance = backend.text_to_image(
                dict(data, parameters=ref_parameters, width=args.size-13, height=args.size), reference)
            assert result.size == (args.size-13, args.size) and result.mode == "RGBA"
            assert provenance["request"]["parameters"]["reference_image_hash"]
            sampling = provenance["runtime"]["provider_runtime"]["sampling"]
            assert sampling["reference_count"] == 1 and sampling["reference_fit"] == "contain"
            if args.turbo:
                assert sampling["sigmas"] == [1., .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0.]
            result.save(args.output / f"reference-{seed}.png")
            (args.output / f"reference-{seed}.json").write_text(json.dumps(provenance, indent=2))
            print("Reference generation:", seed, result.size, flush=True)
        # Switch tools: verify explicit disposal/reload, primary alpha and ref IPC.
        backend.unload_text_to_image()
        source = Image.new("RGBA", (args.size, args.size), (0, 0, 0, 0))
        ImageDraw.Draw(source).ellipse((args.size//4, args.size//4, args.size*3//4, args.size*3//4), fill=(240, 30, 30, 255))
        source.save(args.output / "source.png")
        p["prompt"] = "Change the red circle in <image1> into a blue circle. Keep the background transparent."
        print("Loading image edit", flush=True)
        backend.load_image_edit(data)
        result, seed, provenance = backend.image_edit(data, source, source.copy())
        assert result.size == source.size and result.mode == "RGBA"
        result.save(args.output / "image-edit.png")
        (args.output / "image-edit.json").write_text(json.dumps(provenance, indent=2))
        print("Image edit:", result.size, result.getextrema(), flush=True)
        print(f"Completed in {time.monotonic()-started:.1f}s", flush=True)
    finally:
        backend.unload_text_to_image()
        backend.unload_image_edit()


if __name__ == "__main__":
    main()
