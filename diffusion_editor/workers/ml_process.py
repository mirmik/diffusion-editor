"""Main-process client for the isolated Diffusers/Transformers runtime."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any

import numpy as np
from PIL import Image

from .ml_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    MlProtocolError,
    MlResponse,
    decode_response,
    encode_message,
)


DEFAULT_STARTUP_TIMEOUT = 20.0
DEFAULT_REQUEST_TIMEOUT = 1800.0
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_DETECTIONS = 1024


def default_worker_python() -> Path:
    configured = os.environ.get("DIFFUSION_EDITOR_ML_PYTHON")
    if configured:
        return Path(configured).expanduser().absolute()
    project_root = Path(__file__).resolve().parents[2]
    if os.name == "nt":
        return project_root / ".venv-workers" / "Scripts" / "python.exe"
    return project_root / ".venv-workers" / "bin" / "python"


class MlProcessClient:
    """Owns one persistent, lazily started ML subprocess."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        backend: str = "real",
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._python = Path(python) if python is not None else default_worker_python()
        self._backend = backend
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: Queue[bytes | None] | None = None
        self._reader: threading.Thread | None = None
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._process is not None and self._process.poll() is None

    def request(
        self,
        operation: str,
        data: dict[str, Any],
        cancel: threading.Event,
        *,
        images: dict[str, Image.Image | np.ndarray | None] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        with self._operation_lock:
            process, responses = self._ensure_process()
            request_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(
                prefix="diffusion-editor-ml-"
            ) as directory:
                root = Path(directory)
                payload = dict(data)
                for name, image in (images or {}).items():
                    if image is None:
                        payload[f"{name}_path"] = None
                        continue
                    path = root / f"{name}.png"
                    if isinstance(image, np.ndarray):
                        if image.ndim < 2 or image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
                            raise ValueError(f"ML {name} image exceeds the pixel limit")
                        Image.fromarray(image).save(path, format="PNG")
                    else:
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise ValueError(f"ML {name} image exceeds the pixel limit")
                        image.save(path, format="PNG")
                    payload[f"{name}_path"] = str(path)
                payload["output_dir"] = str(root)
                message = encode_message(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "type": "request",
                        "request_id": request_id,
                        "operation": operation,
                        "data": payload,
                    }
                )
                try:
                    assert process.stdin is not None
                    process.stdin.write(message)
                    process.stdin.flush()
                    while True:
                        response = self._wait_response(
                            process,
                            responses,
                            cancel=cancel,
                            timeout=self._request_timeout,
                        )
                        if response.request_id != request_id:
                            raise MlProtocolError(
                                "ML worker response request ID mismatch"
                            )
                        if response.kind == "progress":
                            if on_progress is not None:
                                on_progress(response.message or "")
                            continue
                        break
                    if response.kind == "error":
                        raise RuntimeError(response.error or "ML worker failed")
                    if response.kind != "result" or response.data is None:
                        raise MlProtocolError(
                            "ML worker returned a non-result response"
                        )
                    return self._materialize_result(root, response.data)
                except Exception:
                    self._stop_process(timeout=0.1)
                    raise

    @staticmethod
    def _materialize_result(
        root: Path,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(data)
        output_path = result.pop("output_path", None)
        if output_path is not None:
            path = Path(output_path)
            if path.parent != root or not path.is_file():
                raise MlProtocolError("ML worker returned an unexpected output path")
            with Image.open(path) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise MlProtocolError("ML result exceeds the pixel limit")
                result["image"] = image.copy()
        detections_path = result.pop("detections_path", None)
        if detections_path is not None:
            path = Path(detections_path)
            if path.parent != root or not path.is_file():
                raise MlProtocolError(
                    "ML worker returned an unexpected detections path"
                )
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise MlProtocolError("ML detections result must be a list")
            if len(raw) > MAX_DETECTIONS:
                raise MlProtocolError("ML result has too many detections")
            detections = []
            for index, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise MlProtocolError("ML detection must be an object")
                detection = dict(item)
                mask_path = detection.pop("mask_path", None)
                if mask_path is not None:
                    candidate = Path(mask_path)
                    if candidate.parent != root or not candidate.is_file():
                        raise MlProtocolError(
                            "ML worker returned an unexpected mask path"
                        )
                    with Image.open(candidate) as mask:
                        if mask.width * mask.height > MAX_IMAGE_PIXELS:
                            raise MlProtocolError(
                                "ML mask exceeds the pixel limit"
                            )
                        detection["mask"] = np.array(
                            mask.convert("L"), dtype=np.uint8
                        ).astype(bool)
                else:
                    detection["mask"] = None
                detections.append(detection)
            result["detections"] = detections
        return result

    def shutdown(self, timeout: float = 1.0) -> None:
        self._stop_process(timeout)

    def _ensure_process(
        self,
    ) -> tuple[subprocess.Popen[bytes], Queue[bytes | None]]:
        with self._state_lock:
            if self._process is not None and self._process.poll() is None:
                assert self._responses is not None
                return self._process, self._responses
            self._stop_process_locked()
            if not self._python.is_file():
                raise RuntimeError(
                    f"Worker Python not found: {self._python}. "
                    "Run ./setup-workers.sh or set "
                    "DIFFUSION_EDITOR_ML_PYTHON."
                )
            project_root = Path(__file__).resolve().parents[2]
            command = [
                str(self._python),
                "-X",
                "gil=1",
                "-u",
                "-m",
                "diffusion_editor.workers.ml_worker",
                "--backend",
                self._backend,
            ]
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=(os.name != "nt"),
            )
            responses: Queue[bytes | None] = Queue(maxsize=8)
            reader = threading.Thread(
                target=self._read_responses,
                args=(process, responses),
                name="ml-worker-reader",
                daemon=True,
            )
            self._process = process
            self._responses = responses
            self._reader = reader
            reader.start()
            try:
                ready = self._wait_response(
                    process,
                    responses,
                    cancel=threading.Event(),
                    timeout=self._startup_timeout,
                )
                self._validate_runtime(ready)
            except Exception:
                self._stop_process_locked()
                raise
            return process, responses

    @staticmethod
    def _read_responses(
        process: subprocess.Popen[bytes],
        responses: Queue[bytes | None],
    ) -> None:
        assert process.stdout is not None
        try:
            while message := process.stdout.readline(MAX_MESSAGE_BYTES + 1):
                responses.put(message)
        finally:
            responses.put(None)

    @staticmethod
    def _wait_response(
        process: subprocess.Popen[bytes],
        responses: Queue[bytes | None],
        *,
        cancel: threading.Event,
        timeout: float,
    ) -> MlResponse:
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                raise RuntimeError("ML operation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"ML worker timed out after {timeout:.1f}s")
            try:
                message = responses.get(timeout=min(0.05, remaining))
            except Empty:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"ML worker exited with code {process.returncode}"
                    )
                continue
            if message is None:
                try:
                    returncode = process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    returncode = process.poll()
                raise RuntimeError(f"ML worker exited with code {returncode}")
            return decode_response(message)

    @staticmethod
    def _validate_runtime(response: MlResponse) -> None:
        if response.kind != "ready" or response.runtime is None:
            raise MlProtocolError("ML worker did not send a ready response")
        runtime = response.runtime
        if runtime.python != "CPython":
            raise MlProtocolError("ML worker must use CPython")
        try:
            major, minor, *_ = map(int, runtime.version.split("."))
        except ValueError as exc:
            raise MlProtocolError(
                "ML worker reported an invalid Python version"
            ) from exc
        if (major, minor) < (3, 11) or major >= 4:
            raise MlProtocolError(
                f"Unsupported ML worker Python: {runtime.version}"
            )
        if "t" in runtime.abiflags and not runtime.gil_enabled:
            raise MlProtocolError("Free-threaded ML worker must enable the GIL")

    def _stop_process(self, timeout: float = 1.0) -> None:
        with self._state_lock:
            self._stop_process_locked(timeout)

    def _stop_process_locked(self, timeout: float = 1.0) -> None:
        process = self._process
        self._process = None
        self._responses = None
        self._reader = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=max(timeout, 0.5))
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=max(timeout, 1.0))
                except subprocess.TimeoutExpired:
                    # A large model process can remain in uninterruptible kernel
                    # cleanup briefly after SIGKILL. Do not mask the operation's
                    # original error or exceed the caller's shutdown budget.
                    threading.Thread(
                        target=process.wait,
                        name="ml-worker-reaper",
                        daemon=True,
                    ).start()
