"""Main-process client for the isolated pose-estimation runtime."""

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

import numpy as np
from PIL import Image
from tcbase import log

from ..generation.pose_estimation import PoseEstimationResult
from .pose_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    PoseProtocolError,
    PoseResponse,
    decode_response,
    encode_message,
)


DEFAULT_STARTUP_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 600.0


def default_worker_python() -> Path:
    configured = os.environ.get("DIFFUSION_EDITOR_POSE_PYTHON")
    if configured:
        return Path(configured).expanduser().absolute()
    project_root = Path(__file__).resolve().parents[2]
    if os.name == "nt":
        return project_root / ".venv-pose" / "Scripts" / "python.exe"
    return project_root / ".venv-pose" / "bin" / "python"


class PoseProcessClient:
    def __init__(
            self,
            *,
            python: str | Path | None = None,
            startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
            request_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        self._python = Path(python) if python is not None else default_worker_python()
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

    def estimate(
            self,
            image: np.ndarray,
            profile_id: str,
            cancel: threading.Event,
            on_progress: Callable[[str], None] | None = None,
    ) -> PoseEstimationResult:
        with self._operation_lock:
            process, responses = self._ensure_process()
            request_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(
                    prefix="diffusion-editor-pose-") as directory:
                root = Path(directory)
                image_path = root / "image.png"
                output_path = root / "pose.json"
                mode = "RGBA" if image.shape[2] == 4 else "RGB"
                Image.fromarray(image, mode).save(image_path, format="PNG")
                request = encode_message({
                    "protocol": PROTOCOL_VERSION,
                    "type": "estimate",
                    "request_id": request_id,
                    "profile_id": profile_id,
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                })
                try:
                    assert process.stdin is not None
                    process.stdin.write(request)
                    process.stdin.flush()
                    while True:
                        response = self._wait_response(
                            process, responses, cancel, self._request_timeout)
                        if response.request_id != request_id:
                            raise PoseProtocolError(
                                "Pose response request ID mismatch")
                        if response.kind == "progress":
                            if on_progress is not None:
                                on_progress(response.message or "")
                            continue
                        break
                    if response.kind == "error":
                        raise RuntimeError(response.error or "Pose worker failed")
                    if response.kind != "result":
                        raise PoseProtocolError(
                            "Pose worker returned a non-result response")
                    if Path(response.output_path or "") != output_path:
                        raise PoseProtocolError(
                            "Pose worker returned an unexpected path")
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    return PoseEstimationResult.from_dict(payload)
                except Exception:
                    self._stop_process(timeout=0.1)
                    raise

    def shutdown(self, timeout: float = 1.0) -> None:
        self._stop_process(timeout)

    def _ensure_process(self):
        with self._state_lock:
            if self._process is not None and self._process.poll() is None:
                assert self._responses is not None
                return self._process, self._responses
            self._stop_process_locked()
            if not self._python.is_file():
                raise RuntimeError(
                    f"Worker Python not found: {self._python}. "
                    "Run ./setup-pose-worker.sh or set "
                    "DIFFUSION_EDITOR_POSE_PYTHON."
                )
            project_root = Path(__file__).resolve().parents[2]
            command = [
                str(self._python), "-X", "gil=1", "-u", "-m",
                "diffusion_editor.workers.pose_worker",
            ]
            log.info(f"[Pose worker] Starting process (python={self._python})")
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                start_new_session=(os.name != "nt"),
            )
            responses: Queue[bytes | None] = Queue(maxsize=4)
            reader = threading.Thread(
                target=self._read_responses,
                args=(process, responses),
                name="pose-worker-reader",
                daemon=True,
            )
            self._process = process
            self._responses = responses
            self._reader = reader
            reader.start()
            try:
                ready = self._wait_response(
                    process, responses, threading.Event(), self._startup_timeout)
                self._validate_runtime(ready)
            except Exception:
                self._stop_process_locked()
                raise
            return process, responses

    @staticmethod
    def _read_responses(process, responses) -> None:
        assert process.stdout is not None
        try:
            while message := process.stdout.readline(MAX_MESSAGE_BYTES + 1):
                responses.put(message)
        finally:
            responses.put(None)

    @staticmethod
    def _wait_response(
            process,
            responses,
            cancel: threading.Event,
            timeout: float) -> PoseResponse:
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                raise RuntimeError("Pose operation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pose worker timed out after {timeout:.1f}s")
            try:
                message = responses.get(timeout=min(0.05, remaining))
            except Empty:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Pose worker exited with code {process.returncode}")
                continue
            if message is None:
                returncode = process.poll()
                raise RuntimeError(f"Pose worker exited with code {returncode}")
            return decode_response(message)

    @staticmethod
    def _validate_runtime(response: PoseResponse) -> None:
        if response.kind != "ready" or response.runtime is None:
            raise PoseProtocolError("Pose worker did not send a ready response")
        runtime = response.runtime
        if runtime.python != "CPython":
            raise PoseProtocolError("Pose worker must use CPython")
        try:
            major, minor, *_ = map(int, runtime.version.split("."))
        except ValueError as exc:
            raise PoseProtocolError(
                "Pose worker reported an invalid Python version") from exc
        if (major, minor) < (3, 11) or major >= 4:
            raise PoseProtocolError(
                f"Unsupported pose worker Python: {runtime.version}")
        if "t" in runtime.abiflags and not runtime.gil_enabled:
            raise PoseProtocolError(
                "Free-threaded pose worker must enable the GIL")

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
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
