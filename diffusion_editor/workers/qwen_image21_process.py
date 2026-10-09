"""Persistent, explicitly owned child process for Qwen Image 2.1."""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import tempfile
import threading

from PIL import Image


class QwenImage21Pipeline:
    scheduler = None

    def __init__(self, parameters, adapters=(), *, check_only=False, timeout=1800, runner_path=None):
        self._process = None
        self._temporary = None
        self._log = None
        self._timeout = timeout
        self._responses = Queue()
        self.parameters = dict(parameters)
        for key in ("runtime_root", "runtime_python", "model",
                    "text_encoder_checkpoint", "vae_checkpoint"):
            path = Path(self.parameters[key]).expanduser().absolute()
            self.parameters[key] = str(path)
            valid = path.is_dir() if key == "runtime_root" else path.is_file()
            if not valid:
                raise FileNotFoundError(f"Qwen Image 2.1 {key} not found: {path}")
        self.adapters = [dict(adapter) for adapter in adapters]
        for adapter in self.adapters:
            if adapter["enabled"] and adapter["source"]:
                path = Path(adapter["source"]).expanduser().absolute()
                if not path.is_file() or path.suffix != ".safetensors":
                    raise ValueError("Qwen 2.1 LoRA must be a local safetensors file")
                adapter["source"] = str(path)
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="diffusion-editor-qwen21-")
            self._directory = Path(self._temporary.name)
            self._log = (self._directory / "runtime.log").open("w+b")
            env = os.environ.copy()
            # The outer worker's Python paths must not override the dedicated env.
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [self.parameters["runtime_python"], "-u",
                 str(runner_path or Path(__file__).with_name("qwen_image21_runner.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, encoding="utf-8", env=env, cwd=self._directory,
            )
            threading.Thread(target=self._read_responses,
                             args=(self._process.stdout, self._responses), daemon=True).start()
            self.runtime_info = self._request({
                "operation": "check" if check_only else "load",
                "parameters": self.parameters, "adapters": self.adapters,
            })
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read_responses(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    def _request(self, request):
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Qwen Image 2.1 process is not running")
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = self._responses.get(timeout=self._timeout)
            if line is None:
                raise RuntimeError("Qwen Image 2.1 process exited unexpectedly")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Qwen Image 2.1 failed"))
            return response["result"]
        except (Empty, OSError, ValueError, RuntimeError) as exc:
            # Include a bounded diagnostic tail, then dispose of broken state.
            self._log.flush()
            with (self._directory / "runtime.log").open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 6000))
                tail = log.read().decode("utf-8", errors="replace")
            message = "Qwen Image 2.1 timed out" if isinstance(exc, Empty) else str(exc)
            self.close()
            raise RuntimeError(f"{message}\n{tail}") from exc

    def generate(self, parameters, seed, width, height, *, image=None, reference_image=None):
        if self._process is None:
            raise RuntimeError("Qwen Image 2.1 process is closed; reload the model")
        images = []
        for i, source in enumerate((image, reference_image)):
            if source is not None:
                path = self._directory / f"input-{i}.png"
                source.save(path, format="PNG")
                images.append(str(path))
        output = self._directory / "result.png"
        output.unlink(missing_ok=True)
        self._request({
            "operation": "generate", "parameters": parameters, "seed": seed,
            "width": width, "height": height, "images": images, "output": str(output),
        })
        with Image.open(output) as result:
            return result.copy()

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
