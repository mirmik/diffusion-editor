"""Persistent settings for a single camera-guided texture pass."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StableGenSettings:
    prompt: str = ''
    negative: str = 'blurry, low quality, text, watermark, harsh shadows'
    reference: str = ''
    seed: int = 31415
    steps: int = 8
    cfg: float = 1.5
    denoise: float = 0.55
    control_strength: float = 0.75
    ip_strength: float = 0.65
    size: int = 768
    checkpoint: str = 'RealVisXL_V5.0_fp16.safetensors'
    prediction_type: str = 'auto'
    sampler: str = 'auto'
    lora: str = 'sdxl_lightning_8step_lora.safetensors'
    depth_model: str = 'controlnet_depth_sdxl.safetensors'
    ip_adapter: str = 'ip-adapter-plus_sdxl_vit-h.safetensors'
    image_encoder: str = 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors'

    def __post_init__(self):
        if self.prediction_type not in ('auto','epsilon','v_prediction'):
            raise ValueError('Invalid StableGen prediction mode')
        if self.sampler not in ('auto','euler','dpmpp_sde_karras'):
            raise ValueError('Invalid StableGen sampler')
        if not 0 <= self.seed <= 2**31-1 or not 1 <= self.steps <= 100:
            raise ValueError('Invalid StableGen seed or steps')
        for name, low, high in [('cfg',0,20),('denoise',0.01,1),('control_strength',0,2),('ip_strength',0,2)]:
            value=getattr(self,name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid StableGen {name}')
        if self.size not in (512,768,1024):
            raise ValueError('StableGen size must be 512, 768 or 1024')
