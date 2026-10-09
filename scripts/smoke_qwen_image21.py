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
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    p = text_to_image_profile("qwen-image-2.1").defaults()
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
    data = {"profile_id": "qwen-image-2.1", "parameters": p, "lora_adapters": []}
    started = time.monotonic()
    try:
        print("Loading Qwen Image 2.1", flush=True)
        info = backend.load_text_to_image(data)
        print(json.dumps(info, indent=2), flush=True)
        result, seed, provenance = backend.text_to_image(dict(data, width=args.size, height=args.size))
        assert result.size == (args.size, args.size) and result.mode == "RGBA"
        result.save(args.output / "text-to-image.png")
        if args.steps >= 30:
            assert result.getextrema()[3][0] < 32, "No transparent background generated"
        (args.output / "text-to-image.json").write_text(json.dumps(provenance, indent=2))
        print("Text-to-image:", result.size, result.getextrema(), flush=True)
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
