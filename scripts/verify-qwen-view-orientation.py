#!/usr/bin/env python3
"""Verify generated Qwen views with a remote multimodal chat model."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.generation.view_orientation import (
    ORIENTATION_PROMPT,
    orientation_verification,
)


DEFAULT_API_BASE = os.environ.get(
    "DIFFUSION_EDITOR_ORIENTATION_VERIFIER_API",
    "http://192.168.0.61:8096/v1",
)
DEFAULT_MODEL = "qwen3.8-27b-uncensored-q4-mtp"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _records(manifest: dict) -> list[dict]:
    records = manifest.get("jobs")
    if not isinstance(records, list):
        records = manifest.get("variants")
    if not isinstance(records, list):
        raise SystemExit("manifest has neither a jobs nor a variants list")
    selected = [
        record for record in records
        if record.get("azimuth_degrees") is not None
        and record.get("output") is not None
    ]
    if not selected:
        raise SystemExit("manifest contains no oriented image records")
    return selected


def _classify(
    *,
    api_base: str,
    model: str,
    image: Image.Image,
    max_new_tokens: int,
    timeout: float,
) -> str:
    encoded_image = BytesIO()
    image.convert("RGB").save(encoded_image, format="PNG")
    data_url = (
        "data:image/png;base64,"
        + base64.b64encode(encoded_image.getvalue()).decode("ascii")
    )
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": ORIENTATION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = api_base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("orientation verifier returned invalid JSON") from exc
    if not isinstance(content, str):
        raise RuntimeError("orientation verifier returned non-text content")
    return content.strip()


def main() -> int:
    args = _arguments()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _records(manifest)
    started = time.monotonic()
    accepted = True
    for index, record in enumerate(records, start=1):
        path = Path(record["output"])
        if not path.is_file():
            raise SystemExit(f"missing generated view: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        response = _classify(
            api_base=args.api_base,
            model=args.model,
            image=image,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        verification = orientation_verification(
            azimuth_degrees=float(record["azimuth_degrees"]),
            response=response,
        )
        record["orientation_verification"] = verification
        accepted = accepted and bool(verification["accepted"])
        print(
            f"[{index}/{len(records)}] {path.name}: "
            f"expected={verification['expected']} "
            f"observed={verification['observed']} "
            f"accepted={verification['accepted']}",
            flush=True,
        )

    manifest["orientation_verification"] = {
        "schema": "diffusion-editor.qwen-view-orientation-verification",
        "schema_version": 1,
        "verifier": args.model,
        "api_base": args.api_base,
        "prompt": ORIENTATION_PROMPT,
        "accepted": accepted,
        "checked_views": len(records),
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        f"Orientation verification {'PASSED' if accepted else 'FAILED'}: "
        f"{manifest_path}",
        flush=True,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
