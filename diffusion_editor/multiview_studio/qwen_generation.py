"""Long-lived Qwen Multiple Angles worker adapter for view slots."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Callable, Iterable

from PIL import Image

from ..generation.image_edit_profiles import (
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from ..workers.ml_process import MlProcessClient
from .model import MultiviewProject, ViewKey


SOURCE_KEYS = (ViewKey("eye", 0), ViewKey("eye", 180))
CARDINAL_GENERATED_KEYS = (ViewKey("eye", 90), ViewKey("eye", 270))
AZIMUTH_PROMPTS = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}
ELEVATION_PROMPTS = {
    "low": "low-angle shot",
    "eye": "eye-level shot",
    "elevated": "elevated shot",
}


def prompt_for_view(key: ViewKey) -> str:
    return (
        f"<sks> {AZIMUTH_PROMPTS[key.azimuth]} "
        f"{ELEVATION_PROMPTS[key.elevation]} medium shot"
    )


def generated_view_keys(
    project: MultiviewProject, mode: str
) -> tuple[ViewKey, ...]:
    if mode == "four":
        return CARDINAL_GENERATED_KEYS
    candidates = tuple(
        slot.key for slot in project.slots if slot.key not in SOURCE_KEYS
    )
    if mode == "all":
        return candidates
    if mode == "missing":
        return tuple(key for key in candidates if not project.slot(key).populated)
    raise ValueError(f"unknown view generation mode: {mode}")


class QwenViewGenerator:
    def __init__(self, client: MlProcessClient | None = None) -> None:
        self._client = client or MlProcessClient()
        self._loaded = False

    def generate(
        self,
        project: MultiviewProject,
        project_path: Path,
        keys: Iterable[ViewKey],
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[ViewKey, str]:
        selected = tuple(dict.fromkeys(keys))
        if not selected:
            return {}
        if any(key in SOURCE_KEYS for key in selected):
            raise ValueError("front/back source slots are not generated")
        front_path = Path(project.front_path)
        back_path = Path(project.back_path)
        if not front_path.is_file() or not back_path.is_file():
            raise ValueError("Front and Back images are required")

        profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
        parameters = profile.defaults()
        parameters.update(seed=project.qwen_seed, steps=4)
        adapters = [
            adapter.to_dict()
            for adapter in (
                *profile.default_lora_adapters,
                qwen_multiple_angles_lora_adapter(),
            )
        ]
        if not self._loaded or not self._client.is_running:
            self._progress(
                on_progress, "Loading Qwen Image Edit + Multiple Angles"
            )
            self._client.request(
                "load_image_edit",
                {
                    "profile_id": profile.stable_id,
                    "parameters": parameters,
                    "lora_adapters": adapters,
                },
                cancel,
                on_progress=lambda message: self._progress(on_progress, message),
            )
            self._loaded = True

        output_dir = project_path.parent / "views"
        output_dir.mkdir(parents=True, exist_ok=True)
        generated: dict[ViewKey, str] = {}
        started = time.monotonic()
        with Image.open(front_path) as front_source:
            front = front_source.convert("RGB")
        with Image.open(back_path) as back_source:
            back = back_source.convert("RGB")
        if back.size != front.size:
            raise ValueError(
                f"Front and Back sizes differ: {front.size} != {back.size}"
            )

        jobs = []
        for index, key in enumerate(selected, 1):
            if cancel.is_set():
                raise RuntimeError("view generation cancelled")
            prompt = prompt_for_view(key)
            self._progress(
                on_progress,
                f"Qwen view {index}/{len(selected)}: {key.stable_id}",
            )
            job_parameters = dict(parameters)
            job_parameters.update(seed=project.qwen_seed, prompt=prompt)
            result = self._client.request(
                "image_edit",
                {
                    "profile_id": profile.stable_id,
                    "parameters": job_parameters,
                    "lora_adapters": adapters,
                },
                cancel,
                images={"image": front, "reference_image": back},
                on_progress=lambda message: self._progress(on_progress, message),
            )
            image = result["image"]
            if image.size != front.size:
                image = image.resize(front.size, Image.Resampling.LANCZOS)
            destination = output_dir / f"mv-{key.stable_id}.png"
            temporary = destination.with_suffix(".tmp.png")
            image.save(temporary, format="PNG")
            temporary.replace(destination)
            generated[key] = str(destination.resolve())
            jobs.append(
                {
                    "view": key.stable_id,
                    "prompt": prompt,
                    "seed": int(result.get("seed", project.qwen_seed)),
                    "image": str(destination.resolve()),
                    "size": list(image.size),
                }
            )

        report = {
            "profile": profile.stable_id,
            "front": str(front_path.resolve()),
            "back": str(back_path.resolve()),
            "seed": project.qwen_seed,
            "steps": 4,
            "jobs": jobs,
            "elapsed_seconds": time.monotonic() - started,
        }
        (output_dir / "generation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return generated

    def shutdown(self, timeout: float = 2.0) -> None:
        self._client.shutdown(timeout)
        self._loaded = False

    @staticmethod
    def _progress(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None and message:
            callback(message)
