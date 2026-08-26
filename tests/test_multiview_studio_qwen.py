from __future__ import annotations

from pathlib import Path
import threading

from PIL import Image

from diffusion_editor.multiview_studio.model import MultiviewProject, ViewKey
from diffusion_editor.multiview_studio.qwen_generation import (
    QwenViewGenerator,
    generated_view_keys,
    prompt_for_view,
)


class _FakeMlClient:
    def __init__(self) -> None:
        self.requests = []
        self.is_running = False

    def request(
        self,
        operation,
        data,
        cancel,
        *,
        images=None,
        on_progress=None,
    ):
        self.requests.append((operation, data, images))
        self.is_running = True
        if on_progress is not None:
            on_progress(operation)
        if operation == "load_image_edit":
            return {"device": "mock", "dtype": "float32"}
        return {
            "image": images["image"].copy(),
            "seed": data["parameters"]["seed"],
        }

    def shutdown(self, _timeout=1.0):
        self.is_running = False


def _project(tmp_path: Path) -> tuple[MultiviewProject, Path]:
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    Image.new("RGB", (64, 96), "red").save(front)
    Image.new("RGB", (64, 96), "blue").save(back)
    project = MultiviewProject().with_source("front", str(front))
    project = project.with_source("back", str(back))
    manifest = tmp_path / "project.mvstudio.json"
    project.save(manifest)
    return project, manifest


def test_group_generation_key_sets_preserve_source_slots(tmp_path: Path):
    project, _manifest = _project(tmp_path)

    assert generated_view_keys(project, "four") == (
        ViewKey("eye", 90),
        ViewKey("eye", 270),
    )
    assert len(generated_view_keys(project, "all")) == 22
    assert ViewKey("eye", 0) not in generated_view_keys(project, "all")
    assert ViewKey("eye", 180) not in generated_view_keys(project, "all")


def test_qwen_generator_uses_exact_angle_prompt_and_shared_seed(tmp_path: Path):
    project, manifest = _project(tmp_path)
    client = _FakeMlClient()
    generator = QwenViewGenerator(client)
    key = ViewKey("elevated", 315)

    generated = generator.generate(
        project,
        manifest,
        (key,),
        threading.Event(),
    )

    assert Path(generated[key]).is_file()
    assert [request[0] for request in client.requests] == [
        "load_image_edit",
        "image_edit",
    ]
    inference = client.requests[-1]
    assert inference[1]["parameters"]["prompt"] == prompt_for_view(key)
    assert inference[1]["parameters"]["seed"] == project.qwen_seed
    assert inference[2]["image"].size == (64, 96)
    assert inference[2]["reference_image"].size == (64, 96)
