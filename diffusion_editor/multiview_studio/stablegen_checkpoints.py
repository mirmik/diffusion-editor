"""Inspect local safetensors headers without loading model tensors."""
import json
import os
from pathlib import Path
import struct


def models_root():
    return Path(os.environ.get('DIFFUSION_EDITOR_STABLEGEN_MODELS','/home/mirmik/soft/ComfyUI/models'))


def is_sdxl_checkpoint(path):
    try:
        with Path(path).open('rb') as file:
            size=struct.unpack('<Q',file.read(8))[0]
            if size>16*1024*1024:return False
            header=json.loads(file.read(size))
        shape=header.get('model.diffusion_model.input_blocks.0.0.weight',{}).get('shape')
        return shape == [320,4,3,3] and any(k.startswith('conditioner.embedders.1.') for k in header)
    except (OSError,ValueError,struct.error):
        return False


def available_checkpoints():
    directory=models_root()/'checkpoints'
    return sorted(str(path.relative_to(directory)) for path in directory.rglob('*.safetensors')
                  if is_sdxl_checkpoint(path))


def resolve_checkpoint(name):
    path=Path(name).expanduser()
    return path if path.is_absolute() else models_root()/'checkpoints'/path
