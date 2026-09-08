"""Cancellable Pixal3D multiview job with persistent inputs, logs and GLB."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import threading
import time

from ..workers.segmentation_process import SegmentationProcessClient
from .model import MultiviewProject
from .pixal3d_views import prepare_views


_LOCAL = Path(__file__).resolve().parents[2] / '.local/pixal3d-multiview'


class Pixal3DGenerator:
    def __init__(self, *, python=None, root=None, model_path=None, segmenter=None):
        self.python = Path(python or os.environ.get('DIFFUSION_EDITOR_PIXAL3D_PYTHON', '/home/mirmik/soft/TRELLIS.2/venv/bin/python')).expanduser()
        self.root = Path(root or os.environ.get('DIFFUSION_EDITOR_PIXAL3D_MV_ROOT', str(_LOCAL / 'upstream') if (_LOCAL / 'upstream/inference_mv.py').exists() else '/home/mirmik/soft/Pixal3D')).expanduser()
        self.model_path = Path(model_path or os.environ.get('DIFFUSION_EDITOR_PIXAL3D_MV_MODEL', str(_LOCAL / 'model') if (_LOCAL / 'model/pipeline_mv.json').exists() else '/home/mirmik/soft/Pixal3D-hf-check')).expanduser()
        self._segmenter = segmenter or SegmentationProcessClient()
        self._process = None
        self._lock = threading.Lock()

    def generate(self, project: MultiviewProject, project_path: Path, cancel: threading.Event, on_progress=None) -> Path:
        return self._generate(project, project_path, cancel, on_progress, operation='shape')

    def generate_texture(self, project, project_path, cancel, on_progress=None):
        return self._generate(project, project_path, cancel, on_progress, operation='texture')

    def _generate(self, project, project_path, cancel, on_progress, *, operation):
        if project.shape_backend != 'pixal3d':
            raise ValueError('Pixal3D generator requires the Pixal3D backend')
        errors = project.validate_texture_request() if operation == 'texture' else project.validate_shape_request()
        if errors:
            raise ValueError('; '.join(errors))
        for path, hint in ((self.python, 'DIFFUSION_EDITOR_PIXAL3D_PYTHON'),
                           (self.root / 'inference_mv.py', 'DIFFUSION_EDITOR_PIXAL3D_MV_ROOT'),
                           (self.model_path / 'pipeline_mv.json', 'DIFFUSION_EDITOR_PIXAL3D_MV_MODEL')):
            if not path.is_file():
                raise FileNotFoundError(f'Pixal3D runtime file missing: {path}. Configure {hint}.')
        if cancel.is_set():
            raise RuntimeError('Pixal3D generation cancelled')
        runs = project_path.resolve().parent / ('texture-runs' if operation == 'texture' else 'shape-runs')
        runs.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f'pixal3d-{time.strftime("%Y%m%d-%H%M%S")}-', dir=runs))
        try:
            prepare_views(project.slots, output / 'views', project.pixal3d, cancel, self._segmenter.segment, on_progress)
        finally:
            # Free segmentation resources before the large GPU pipeline starts.
            self._segmenter.shutdown()
        request = {'protocol': 1, 'backend': 'pixal3d', 'project': str(project_path.resolve()),
                   'root': str(self.root.resolve()), 'model_path': str(self.model_path.resolve()),
                   'settings': asdict(project.pixal3d), 'views_dir': 'views'}
        request['operation'] = operation
        if operation == 'texture':
            encoder = Path(os.environ.get('DIFFUSION_EDITOR_PIXAL3D_SHAPE_ENCODER',
                '/home/mirmik/soft/TRELLIS.2/models/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')).expanduser()
            for suffix in ('.json', '.safetensors'):
                if not Path(str(encoder) + suffix).is_file():
                    raise FileNotFoundError(f'Missing shape encoder: {encoder}{suffix}; configure DIFFUSION_EDITOR_PIXAL3D_SHAPE_ENCODER')
            shutil.copyfile(Path(project.geometry_path).expanduser().resolve(), output / 'input.glb')
            request.update(input_mesh='input.glb', shape_encoder=str(encoder.resolve()),
                           texture_settings=asdict(project.pixal3d_texture))
        request_path = output / 'request.json'
        request_path.write_text(json.dumps(request, indent=2) + '\n')
        runner = Path(__file__).with_name('pixal3d_runner.py')
        return self._run_worker([str(self.python), '-u', str(runner), str(request_path)], output, cancel, on_progress)

    def _run_worker(self, command, output: Path, cancel: threading.Event, on_progress=None) -> Path:
        log_path = output / 'worker.log'
        process = None
        try:
            with log_path.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, cwd=self.root)
                with self._lock:
                    self._process = process
                # Poll the process, not a blocking stdout.readline(): cancellation
                # must work during silent checkpoint loading and CUDA compilation.
                with log_path.open(encoding='utf-8', errors='replace') as reader:
                    while process.poll() is None:
                        if cancel.wait(0.1):
                            raise RuntimeError(f'Pixal3D generation cancelled; log: {log_path}')
                        chunk = reader.read()
                        if chunk.strip() and on_progress:
                            on_progress(chunk.replace('\r', '\n').strip().splitlines()[-1][-400:])
            if cancel.is_set():
                raise RuntimeError(f'Pixal3D generation cancelled; log: {log_path}')
            if process.returncode:
                with log_path.open('rb') as stream:
                    stream.seek(max(0, log_path.stat().st_size - 4000))
                    tail = stream.read().decode('utf-8', errors='replace')
                raise RuntimeError(f'Pixal3D worker exited with code {process.returncode}:\n{tail}\nLog: {log_path}')
            result_path = output / 'result.json'
            if not result_path.is_file():
                raise RuntimeError(f'Pixal3D worker produced no result; log: {log_path}')
            result = json.loads(result_path.read_text())
            if result.get('status') != 'success':
                raise RuntimeError(f'Pixal3D failed: {result.get("error", "unknown error")}; log: {log_path}')
            shape = output / result['shape']
            if not shape.is_file() or not shape.stat().st_size:
                raise RuntimeError(f'Pixal3D output missing: {shape}; log: {log_path}')
            return shape.resolve()
        finally:
            if process is not None:
                self._terminate(process)
            with self._lock:
                if self._process is process:
                    self._process = None

    def cancel(self):
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
