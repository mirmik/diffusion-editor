import os
import subprocess
import sys

from diffusion_editor.app.application import (
    EditorApplication,
    EngineSet,
    ShutdownPhase,
)
from diffusion_editor.app.presentation import HeadlessEditorPresentation


class _MemorySettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _Engine:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.model_info = {}

    def shutdown(self):
        self.calls.append(self.name)

    def poll_event(self):
        return None


def _application(calls=None):
    calls = calls if calls is not None else []
    engines = EngineSet(
        diffusion=_Engine("diffusion", calls),
        segmentation=_Engine("segmentation", calls),
        lama=_Engine("lama", calls),
        instruct=_Engine("instruct", calls),
        grounding=_Engine("grounding", calls),
    )
    return EditorApplication(settings=_MemorySettings(), engines=engines)


def test_application_import_and_construction_do_not_load_tcgui():
    code = """
import sys
from diffusion_editor.app.application import EditorApplication, EngineSet

class Settings:
    def get(self, key, default=None): return default
    def set(self, key, value): pass

class Engine:
    model_info = {}
    def shutdown(self): pass

engine = Engine()
EditorApplication(
    settings=Settings(),
    engines=EngineSet(engine, engine, engine, engine, engine),
)
assert not any(name == 'tcgui' or name.startswith('tcgui.') for name in sys.modules)
"""
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_headless_presentation_binds_to_application_owner():
    application = _application()
    presentation = HeadlessEditorPresentation()

    application.bind_view(presentation.ports())
    application.set_status("Bound")
    application.update_panel("layers", "ready", count=3)

    assert presentation.status == "Bound"
    assert presentation.panel_updates[-1].panel_id == "layers"
    assert presentation.panel_updates[-1].payload == {"count": 3}


def test_shutdown_orders_view_workers_engines_and_gpu_resources():
    calls = []
    application = _application(calls)
    application.register_shutdown_resource(
        ShutdownPhase.GPU_RESOURCES,
        "canvas",
        lambda: calls.append("canvas"),
    )
    application.register_shutdown_resource(
        ShutdownPhase.VIEW_WORKERS,
        "chat",
        lambda: calls.append("chat"),
    )

    application.close()
    application.close()

    assert calls == [
        "chat",
        "diffusion",
        "instruct",
        "lama",
        "segmentation",
        "grounding",
        "canvas",
    ]
    assert application.shutdown_trace == [
        "chat",
        "diffusion-engine",
        "instruct-engine",
        "lama-engine",
        "segmentation-engine",
        "grounding-engine",
        "canvas",
    ]
