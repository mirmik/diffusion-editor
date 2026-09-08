"""Cancellable one-shot local GPU jobs for camera texture passes."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import sys
import uuid



def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StableGenService:
    def __init__(self, python=None):
        self.python=python or os.environ.get('DIFFUSION_EDITOR_STABLEGEN_PYTHON','/home/mirmik/soft/TRELLIS.2/venv/bin/python')
        self.process=None

    def gpu(self, root, operation, cancel):
        if operation not in ('prepare', 'generate', 'project'):
            raise ValueError(f'Unknown StableGen operation: {operation}')
        if cancel.is_set():
            raise RuntimeError('StableGen pass cancelled')
        runner=Path(__file__).with_name(
            'stablegen_image_runner.py' if operation == 'generate' else 'stablegen_projection_runner.py')
        with (root/f'{operation}.log').open('w') as log:
            process=subprocess.Popen([str(self.python),'-u',str(runner),str(root),operation],stdout=log,stderr=subprocess.STDOUT)
            self.process=process
            try:
                while process.poll() is None:
                    if cancel.wait(.1):
                        process.terminate()
                        try: process.wait(3)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                        raise RuntimeError('StableGen pass cancelled')
                if process.returncode:
                    raise RuntimeError((root/f'{operation}.log').read_text()[-5000:])
            finally:
                self.process=None
        if cancel.is_set(): raise RuntimeError('StableGen pass cancelled')

    def prepare(self, shape, project_path, camera, settings, cancel):
        root=project_path.parent/'texture-runs'/('stablegen-'+uuid.uuid4().hex[:12])
        root.mkdir(parents=True)
        shutil.copyfile(shape,root/'input.glb')
        request=dict(protocol=1,camera=camera,source_sha256=fingerprint(shape),settings=asdict(settings))
        (root/'request.json').write_text(json.dumps(request,indent=2))
        self.gpu(root,'prepare',cancel)
        return root

    def generate(self, root, settings, cancel):
        from PIL import Image
        if not Path(settings.reference).expanduser().is_file(): raise ValueError('Choose an existing reference image for IPAdapter')
        with Image.open(Path(settings.reference).expanduser()) as image: image.convert('RGB').save(root/'reference.png')
        request=json.loads((root/'request.json').read_text());request['settings']=asdict(settings)
        request['reference_sha256']=fingerprint(root/'reference.png')
        (root/'request.json').write_text(json.dumps(request,indent=2))
        for name in ('candidate.png', 'candidate.glb', 'generation.json'):
            (root/name).unlink(missing_ok=True)
        self.gpu(root,'generate',cancel)
        if not (root/'candidate.png').is_file():
            raise RuntimeError('StableGen image worker completed without a candidate')
        return root/'candidate.png'


    def project(self, root, cancel):
        self.validate_image(root, root/'candidate.png')
        (root/'candidate.glb').unlink(missing_ok=True)
        self.gpu(root, 'project', cancel)
        return root/'candidate.glb'

    @staticmethod
    def validate_image(root, path):
        from PIL import Image
        with Image.open(root/'input-rgb.png') as source, Image.open(path) as image:
            if image.size != source.size:
                raise ValueError(f'Image must keep captured camera dimensions {source.size}; got {image.size}')
            image.load()

    def import_image(self, root, path):
        from PIL import Image
        self.validate_image(root, path)
        with Image.open(path) as image:
            image.convert('RGB').save(root/'candidate-import.png')
        (root/'candidate-import.png').replace(root/'candidate.png')
        (root/'candidate.glb').unlink(missing_ok=True)
        return root/'candidate.png'

    def edit_image(self, root, cancel):
        if cancel.is_set(): raise RuntimeError('Image editing cancelled')
        for name in ('editor-return.json', 'editor-cancel'):
            (root/name).unlink(missing_ok=True)
        source=root/('candidate.png' if (root/'candidate.png').is_file() else 'input-rgb.png')
        shutil.copyfile(source, root/'editor-input.png')
        env=dict(os.environ)
        package_root=str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH']=os.pathsep.join(filter(None,[package_root,env.get('PYTHONPATH')]))
        with (root/'editor.log').open('w') as log:
            process=subprocess.Popen(
                [sys.executable,'-m','diffusion_editor.multiview_studio.image_editor_bridge',str(root)],
                stdout=log,stderr=subprocess.STDOUT,env=env)
            self.process=process
            try:
                while process.poll() is None:
                    if cancel.wait(.1):
                        # Let EditorApplication close its own model workers first.
                        (root/'editor-cancel').touch()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try: process.wait(timeout=3)
                            except subprocess.TimeoutExpired: process.kill();process.wait()
                        raise RuntimeError('Image editing cancelled')
                if process.returncode:
                    raise RuntimeError((root/'editor.log').read_text()[-5000:])
            finally:
                self.process=None
        if cancel.is_set(): raise RuntimeError('Image editing cancelled')
        if not (root/'editor-return.json').is_file(): return None
        return self.import_image(root, root/'editor-result.png')
