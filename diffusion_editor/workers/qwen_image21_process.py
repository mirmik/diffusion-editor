"""Persistent, explicitly owned child process for Qwen Image 2.1."""
from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading

from PIL import Image
from ..generation.progress import GenerationProgress


class QwenImage21Pipeline:
    scheduler = None

    def __init__(self, parameters, adapters=(), *, check_only=False, timeout=1800, runner_path=None):
        self._process = None
        self._temporary = None
        self._log = None
        self._stderr_thread = None
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
        if self.parameters.get("sampling_mode") == "turbo8":
            turbo = [a for a in self.adapters if a.get("stable_id") == "turbo8"]
            if (len(turbo) != 1 or not turbo[0]["enabled"] or not turbo[0]["source"]
                    or float(turbo[0]["weight"]) != 1.0):
                raise ValueError("Turbo8 requires its enabled LoRA at strength 1.0")
        for adapter in self.adapters:
            if adapter["enabled"] and adapter["source"]:
                path = Path(adapter["source"]).expanduser().absolute()
                if not path.is_file() or path.suffix != ".safetensors":
                    raise ValueError("Qwen 2.1 LoRA must be a local safetensors file")
                adapter["source"] = str(path)
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="diffusion-editor-qwen21-")
            self._directory = Path(self._temporary.name)
            self._log = (self._directory / "runtime.log").open("wb", buffering=0)
            env = os.environ.copy()
            # The outer worker's Python paths must not override the dedicated env.
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [self.parameters["runtime_python"], "-u",
                 str(runner_path or Path(__file__).with_name("qwen_image21_runner.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", env=env, cwd=self._directory,
            )
            threading.Thread(target=self._read_responses,
                             args=(self._process.stdout, self._responses), daemon=True).start()
            self._stderr_thread = threading.Thread(
                target=self._relay_stderr,
                args=(self._process.stderr, self._log, sys.stderr), daemon=True,
            )
            self._stderr_thread.start()
            self.runtime_info = self._request({
                "operation": "check" if check_only else "load",
                "parameters": self.parameters, "adapters": self.adapters,
            })
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _relay_stderr(stream, log, terminal):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            # tqdm updates have carriage returns, not newlines. Read bytes to
            # forward each update immediately and preserve split UTF-8 characters.
            chunk = os.read(stream.fileno(), 4096)
            if chunk:
                log.write(chunk)
            text = decoder.decode(chunk, final=not chunk)
            if terminal is not None and text:
                try:
                    terminal.write(text)
                    terminal.flush()
                except (OSError, ValueError, UnicodeError):
                    # A closed/unavailable terminal must not block the runtime.
                    terminal = None
            if not chunk:
                break

    @staticmethod
    def _read_responses(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    def _request(self, request, on_progress=None):
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Qwen Image 2.1 process is not running")
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            while True:
                line = self._responses.get(timeout=self._timeout)
                if line is None:
                    raise RuntimeError("Qwen Image 2.1 process exited unexpectedly")
                response = json.loads(line)
                if response.get("type") != "progress":
                    break
                update = GenerationProgress.from_dict(response)
                if on_progress is not None:
                    on_progress(update)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Qwen Image 2.1 failed"))
            return response["result"]
        except (Empty, OSError, ValueError, RuntimeError) as exc:
            # Include a bounded diagnostic tail, then dispose of broken state.
            self._stop_process()
            with (self._directory / "runtime.log").open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 6000))
                tail = log.read().decode("utf-8", errors="replace")
            message = "Qwen Image 2.1 timed out" if isinstance(exc, Empty) else str(exc)
            self.close()
            raise RuntimeError(f"{message}\n{tail}") from exc

    def generate(self, parameters, seed, width, height, *, image=None, reference_image=None,
                 reference_images=None, on_progress=None):
        if self._process is None:
            raise RuntimeError("Qwen Image 2.1 process is closed; reload the model")
        if reference_images is not None:
            if image is not None or reference_image is not None:
                raise ValueError("Cannot combine reference_images with legacy editing inputs")
            sources = tuple(reference_images)
        else:
            sources = tuple(source for source in (image, reference_image) if source is not None)
        if len(sources) > 10:
            raise ValueError("Qwen Image 2.1 accepts at most 10 reference images")
        images = []
        for i, source in enumerate(sources):
            if source is not None:
                path = self._directory / f"input-{i}.png"
                source.save(path, format="PNG")
                images.append(str(path))
        output = self._directory / "result.png"
        output.unlink(missing_ok=True)
        response = self._request({
            "operation": "generate", "parameters": parameters, "seed": seed,
            "width": width, "height": height, "images": images, "output": str(output),
            "reference_only": image is None,
            "report_progress": on_progress is not None,
        }, on_progress=on_progress)
        if "sampling" in response:
            self.runtime_info["sampling"] = response["sampling"]
        with Image.open(output) as result:
            return result.copy()

    def _stop_process(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def close(self):
        self._stop_process()
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
