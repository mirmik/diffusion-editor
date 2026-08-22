from __future__ import annotations

import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.engines.diffusion_engine import DiffusionEngine
from diffusion_editor.engines.grounding_engine import GroundingEngine
from diffusion_editor.engines.instruct_engine import InstructEngine
from diffusion_editor.generation.types import (
    DiffusionInferenceResult,
    DiffusionRequest,
    InstructInferenceResult,
    InstructRequest,
    ImageEditRequest,
)
from diffusion_editor.generation.image_edit_profiles import (
    FLUX2_KLEIN_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.grounding.types import GroundingParams, GroundingRequest
from diffusion_editor.workers.ml_process import MlProcessClient
from diffusion_editor.workers.ml_protocol import MlProtocolError


def _client(backend: str = "fake", timeout: float = 2.0) -> MlProcessClient:
    return MlProcessClient(
        python=sys.executable,
        backend=backend,
        startup_timeout=timeout,
        request_timeout=timeout,
    )


def _image(color: str = "red") -> Image.Image:
    return Image.new("RGB", (8, 6), color)


def _diffusion_data(mode: str = "img2img") -> dict:
    return {
        "prompt": "test",
        "negative_prompt": "",
        "strength": 0.5,
        "steps": 2,
        "guidance_scale": 1.0,
        "seed": -1,
        "mode": mode,
        "masked_content": "original",
        "ip_adapter_scale": 0.6,
        "width": 8,
        "height": 6,
    }


def _grounding_data() -> dict:
    return {
        "prompt": "object",
        "model_id": "fake/dino",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "use_gpu": False,
        "sam2_model_id": "fake/sam",
        "sam2_mask_channel": 0,
        "mask_threshold": 0.0,
        "max_hole_area": 0,
        "max_sprinkle_area": 0,
        "multimask": False,
        "non_overlap": False,
    }


def _poll(engine, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    event = None
    while event is None and time.monotonic() < deadline:
        event = engine.poll_event()
        time.sleep(0.001)
    assert event is not None
    return event


def test_fake_worker_smokes_model_families_without_main_imports():
    client = _client()
    before = sys._is_gil_enabled()
    progress: list[str] = []
    try:
        loaded = client.request(
            "load_diffusion",
            {"model_path": "fake.safetensors", "prediction_type": None},
            threading.Event(),
            on_progress=progress.append,
        )
        assert loaded["model_path"] == "fake.safetensors"
        diffusion = client.request(
            "diffusion",
            _diffusion_data(),
            threading.Event(),
            images={"image": _image(), "mask": None, "ip_adapter": None},
        )
        assert diffusion["image"].size == (8, 6)
        assert diffusion["seed"] == 4242
        assert diffusion["provenance"]["schema_version"] == 1
        assert diffusion["provenance"]["request"]["kind"] == "diffusion"
        assert (
            diffusion["provenance"]["model"]["status"]
            == "unknown"
        )

        client.request("load_instruct", {}, threading.Event())
        instruct = client.request(
            "instruct",
            {
                "instruction": "make blue",
                "guidance_scale": 7.5,
                "image_guidance_scale": 1.5,
                "steps": 2,
                "seed": -1,
            },
            threading.Event(),
            images={"image": _image()},
        )
        assert instruct["seed"] == 4343
        assert instruct["provenance"]["operation"] == "instruct"
        assert (
            instruct["provenance"]["model"]["status"]
            == "floating"
        )

        grounding = client.request(
            "grounding",
            _grounding_data(),
            threading.Event(),
            images={"image": np.asarray(_image().convert("RGBA"))},
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
            threading.Event(),
            images={"image": _image()},
        )
        depth_array = depth["depth"]
        assert depth_array.shape == (6, 8)
        assert depth_array.dtype == np.float32
        assert depth_array[0, 0] == pytest.approx(1.0)
        assert depth_array[0, -1] == pytest.approx(2.0)
        assert depth["value_kind"] == "inverse_relative"
        assert depth["intrinsics"] is None
        assert grounding["detections"][0]["label"] == "object"
        assert grounding["detections"][0]["mask"].shape == (6, 8)
        assert client.is_running
        assert sys._is_gil_enabled() is before is False
        for module in ("torch", "diffusers", "transformers", "tokenizers"):
            assert module not in sys.modules
    finally:
        client.shutdown()
    assert not client.is_running


def test_depth_float_payload_round_trips_exact_values_and_camera(tmp_path):
    depth = np.array([
        [0.12345679, 12.75],
        [1024.5, 1.0e-5],
    ], dtype="<f4")
    confidence = np.array([
        [0.25, 0.5],
        [0.75, 1.0],
    ], dtype="<f4")
    depth_path = tmp_path / "depth.f32"
    confidence_path = tmp_path / "confidence.f32"
    depth.tofile(depth_path)
    confidence.tofile(confidence_path)
    intrinsics = [
        [777.25, 0.0, 1.125],
        [0.0, 778.5, 0.875],
        [0.0, 0.0, 1.0],
    ]

    result = MlProcessClient._materialize_result(tmp_path, {
        "depth_path": str(depth_path),
        "depth_shape": [2, 2],
        "depth_dtype": "float32-le",
        "confidence_path": str(confidence_path),
        "intrinsics": intrinsics,
        "value_kind": "direct_metric",
        "field_of_view_degrees": 54.125,
        "scale_factor": 1.0,
    })

    assert result["depth"].dtype == np.float32
    np.testing.assert_array_equal(result["depth"], depth)
    np.testing.assert_array_equal(result["confidence"], confidence)
    np.testing.assert_array_equal(
        result["intrinsics"], np.asarray(intrinsics, dtype=np.float32))


@pytest.mark.parametrize(("values", "shape", "error"), (
    (np.array([1.0], dtype="<f4"), [1, 2], "payload size"),
    (np.array([np.nan], dtype="<f4"), [1, 1], "non-finite"),
))
def test_depth_float_payload_rejects_corrupt_data(
        tmp_path, values, shape, error):
    path = tmp_path / "depth.f32"
    values.tofile(path)

    with pytest.raises(MlProtocolError, match=error):
        MlProcessClient._materialize_result(tmp_path, {
            "depth_path": str(path),
            "depth_shape": shape,
            "depth_dtype": "float32-le",
            "confidence_path": None,
            "intrinsics": None,
            "value_kind": "direct_scale_ambiguous",
        })


@pytest.mark.parametrize(
    "profile_id",
    [
        QWEN_IMAGE_EDIT_PROFILE_ID,
        FLUX2_KLEIN_PROFILE_ID,
        SENSENOVA_U15_PROFILE_ID,
    ],
)
def test_fake_worker_smokes_image_edit_profiles(profile_id):
    client = _client()
    profile = image_edit_profile(profile_id)
    parameters = profile.defaults()
    lora_adapters = [
        adapter.to_dict() for adapter in profile.default_lora_adapters]
    parameters.update({"prompt": "change only the body", "seed": 20260820})
    try:
        loaded = client.request(
            "load_image_edit",
            {
                "profile_id": profile_id,
                "parameters": parameters,
                "lora_adapters": lora_adapters,
            },
            threading.Event(),
        )
        assert loaded["profile_id"] == profile_id
        result = client.request(
            "image_edit",
            {
                "profile_id": profile_id,
                "parameters": parameters,
                "lora_adapters": lora_adapters,
            },
            threading.Event(),
            images={"image": _image()},
        )
        assert result["image"].size == (8, 6)
        assert result["seed"] == 20260820
        assert result["provenance"]["operation"] == "image_edit"
        assert (
            result["provenance"]["request"]["parameters"]
            ["model_profile_id"] == profile_id
        )
        assert (
            result["provenance"]["request"]["parameters"]
            ["lora_adapters"] == [
                adapter.to_dict() for adapter in profile.default_lora_adapters
            ]
        )
    finally:
        client.shutdown()


def test_image_edit_engine_reloads_only_for_load_time_parameters():
    engine = InstructEngine(client=_client())
    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    adapters = tuple(
        adapter.to_dict() for adapter in profile.default_lora_adapters)
    try:
        assert engine.submit_load(
            QWEN_IMAGE_EDIT_PROFILE_ID, parameters, adapters)
        assert _poll(engine).error is None
        assert engine.loaded_configuration_matches(
            QWEN_IMAGE_EDIT_PROFILE_ID, parameters, adapters)
        changed_prompt = {**parameters, "prompt": "another edit"}
        assert engine.loaded_configuration_matches(
            QWEN_IMAGE_EDIT_PROFILE_ID, changed_prompt, adapters)
        changed_dtype = {**parameters, "dtype": "float16"}
        assert not engine.loaded_configuration_matches(
            QWEN_IMAGE_EDIT_PROFILE_ID, changed_dtype, adapters)
        changed_adapters = [dict(adapter) for adapter in adapters]
        changed_adapters[0]["weight"] = 0.65
        assert not engine.loaded_configuration_matches(
            QWEN_IMAGE_EDIT_PROFILE_ID, changed_prompt, changed_adapters)
        assert engine.submit_request(ImageEditRequest(
            _image(), QWEN_IMAGE_EDIT_PROFILE_ID, changed_prompt, adapters))
        assert _poll(engine).result.seed == 4343
    finally:
        engine.shutdown()


def test_image_edit_engine_sends_optional_second_image_to_worker():
    class CaptureClient:
        is_running = True

        def request(self, operation, data, cancel, images=None):
            assert operation == "image_edit"
            assert images["image"].getpixel((0, 0)) == (255, 0, 0)
            assert images["reference_image"].getpixel((0, 0)) == (0, 0, 255)
            return {"image": _image("green"), "seed": 17}

    parameters = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID).defaults()
    engine = InstructEngine(client=CaptureClient())
    result = engine._run_inference(ImageEditRequest(
        image=_image("red"),
        model_profile_id=QWEN_IMAGE_EDIT_PROFILE_ID,
        parameters=parameters,
        reference_image=_image("blue"),
    ), threading.Event())

    assert result.seed == 17
    assert result.image.getpixel((0, 0)) == (0, 128, 0)


@pytest.mark.parametrize(
    ("backend", "error"),
    [("crash", "exited with code 39"), ("malformed", "malformed JSON")],
)
def test_ml_crash_and_malformed_response_are_restartable(backend, error):
    client = _client(backend)
    with pytest.raises((RuntimeError, MlProtocolError), match=error):
        client.request("gpu_available", {}, threading.Event())
    assert not client.is_running
    client._backend = "fake"
    try:
        assert client.request(
            "gpu_available", {}, threading.Event()
        ) == {"available": False}
    finally:
        client.shutdown()


def test_ml_cancel_and_timeout_stop_hung_process():
    client = _client("hang", timeout=5.0)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            client.request("gpu_available", {}, cancel)
    finally:
        timer.cancel()
        client.shutdown()
    assert not client.is_running

    client = _client("hang", timeout=0.15)
    with pytest.raises(TimeoutError, match="timed out"):
        client.request("gpu_available", {}, threading.Event())
    assert not client.is_running


def test_diffusion_engine_maps_process_result_and_state():
    engine = DiffusionEngine(client=_client())
    try:
        assert engine.submit_load("fake.safetensors")
        load = _poll(engine)
        assert load.result == "fake.safetensors"
        assert engine.is_loaded

        request = DiffusionRequest(
            image=_image(),
            mask_image=None,
            prompt="test",
            negative_prompt="",
            strength=0.5,
            steps=2,
            guidance_scale=1.0,
            seed=-1,
            mode="img2img",
            masked_content="original",
            ip_adapter_image=None,
            ip_adapter_scale=0.6,
            width=8,
            height=6,
        )
        assert engine.submit_request(request)
        event = _poll(engine)
        assert isinstance(event.result, DiffusionInferenceResult)
        assert event.result.seed == 4242
        assert event.result.provenance is not None
        assert event.result.provenance.operation == "diffusion"
    finally:
        engine.shutdown()


def test_diffusion_engine_clears_loaded_state_after_worker_restart():
    client = _client()
    engine = DiffusionEngine(client=client)
    try:
        assert engine.submit_load("fake.safetensors")
        assert _poll(engine).error is None
        assert engine.model_path == "fake.safetensors"

        # Simulate an externally lost worker. The restarted fake backend has no
        # model, so inference fails and the facade must forget stale model state.
        client.shutdown()
        request = DiffusionRequest(
            image=_image(),
            mask_image=None,
            prompt="test",
            negative_prompt="",
            strength=0.5,
            steps=2,
            guidance_scale=1.0,
            seed=1,
            mode="img2img",
            masked_content="original",
            ip_adapter_image=None,
            ip_adapter_scale=0.6,
            width=8,
            height=6,
        )
        assert engine.submit_request(request)
        assert _poll(engine).error is not None
        assert engine.model_path is None
        assert not engine.is_loaded
    finally:
        engine.shutdown()


def test_instruct_and_grounding_engines_map_process_results():
    instruct = InstructEngine(client=_client())
    try:
        assert instruct.submit_load()
        assert _poll(instruct).error is None
        assert instruct.submit_request(
            InstructRequest(_image(), "test", 7.5, 1.5, 2, -1)
        )
        event = _poll(instruct)
        assert isinstance(event.result, InstructInferenceResult)
        assert event.result.seed == 4343
        assert event.result.provenance is not None
        assert event.result.provenance.operation == "instruct"
    finally:
        instruct.shutdown()

    grounding = GroundingEngine(client=_client())
    params = GroundingParams(**_grounding_data())
    try:
        assert grounding.submit_request(
            GroundingRequest(
                image=np.asarray(_image().convert("RGBA")),
                params=params,
            )
        )
        result_event = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            event = grounding.poll_event()
            if event is not None and event.result is not None:
                result_event = event
                break
            time.sleep(0.001)
        assert result_event is not None
        assert result_event.result.detections[0].label == "object"
        assert result_event.result.canvas_width == 8
        assert result_event.result.canvas_height == 6
    finally:
        grounding.shutdown()
