"""Main-process client for the isolated LaMa runtime."""

from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from PIL import Image

from .lama_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    LamaProtocolError,
    LamaResponse,
    decode_response,
    encode_message,
)


DEFAULT_STARTUP_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 300.0


def default_worker_python() -> Path:
    configured = os.environ.get("DIFFUSION_EDITOR_LAMA_PYTHON")
    if configured:
        return Path(configured).expanduser().absolute()
    project_root = Path(__file__).resolve().parents[2]
    if os.name == "nt":
        return project_root / ".venv-workers" / "Scripts" / "python.exe"
    return project_root / ".venv-workers" / "bin" / "python"


class LamaProcessClient:
    """Owns one persistent, lazily started LaMa subprocess."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        backend: str = "lama",
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

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        cancel: threading.Event,
    ) -> Image.Image:
        with self._operation_lock:
            process, responses = self._ensure_process()
            request_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(
                prefix="diffusion-editor-lama-"
            ) as directory:
                root = Path(directory)
                image_path = root / "image.png"
                mask_path = root / "mask.png"
                output_path = root / "result.png"
                image.convert("RGB").save(image_path, format="PNG")
                mask.convert("L").save(mask_path, format="PNG")
                request = encode_message(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "type": "inpaint",
                        "request_id": request_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "output_path": str(output_path),
                    }
                )
                try:
                    assert process.stdin is not None
                    process.stdin.write(request)
                    process.stdin.flush()
                    response = self._wait_response(
                        process,
                        responses,
                        cancel=cancel,
                        timeout=self._request_timeout,
                    )
                    if response.request_id != request_id:
                        raise LamaProtocolError(
                            "LaMa worker response request ID mismatch"
                        )
                    if response.kind == "error":
                        raise RuntimeError(response.error or "LaMa worker failed")
                    if response.kind != "result":
                        raise LamaProtocolError(
                            "LaMa worker returned a non-result response"
                        )
                    if Path(response.output_path or "") != output_path:
                        raise LamaProtocolError(
                            "LaMa worker returned an unexpected output path"
                        )
                    with Image.open(output_path) as result:
                        return result.convert("RGB").copy()
                except Exception:
                    # Failure paths must not spend the engine's whole shutdown
                    # budget waiting for a worker that is already suspect.
                    self._stop_process(timeout=0.1)
                    raise

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
                    "DIFFUSION_EDITOR_LAMA_PYTHON."
                )
            project_root = Path(__file__).resolve().parents[2]
            command = [
                str(self._python),
                "-X",
                "gil=1",
                "-u",
                "-m",
                "diffusion_editor.workers.lama_worker",
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
            responses: Queue[bytes | None] = Queue(maxsize=4)
            reader = threading.Thread(
                target=self._read_responses,
                args=(process, responses),
                name="lama-worker-reader",
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
    ) -> LamaResponse:
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                raise RuntimeError("LaMa operation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"LaMa worker timed out after {timeout:.1f}s"
                )
            try:
                message = responses.get(timeout=min(0.05, remaining))
            except Empty:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"LaMa worker exited with code {process.returncode}"
                    )
                continue
            if message is None:
                try:
                    returncode = process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    returncode = process.poll()
                raise RuntimeError(
                    f"LaMa worker exited with code {returncode}"
                )
            return decode_response(message)

    @staticmethod
    def _validate_runtime(response: LamaResponse) -> None:
        if response.kind != "ready" or response.runtime is None:
            raise LamaProtocolError("LaMa worker did not send a ready response")
        runtime = response.runtime
        if runtime.python != "CPython":
            raise LamaProtocolError("LaMa worker must use CPython")
        try:
            major, minor, *_ = map(int, runtime.version.split("."))
        except ValueError as exc:
            raise LamaProtocolError(
                "LaMa worker reported an invalid Python version"
            ) from exc
        if (major, minor) < (3, 10) or major >= 4:
            raise LamaProtocolError(
                f"Unsupported LaMa worker Python: {runtime.version}"
            )
        if "t" in runtime.abiflags and not runtime.gil_enabled:
            raise LamaProtocolError(
                "Free-threaded LaMa worker must start with the GIL enabled"
            )

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
