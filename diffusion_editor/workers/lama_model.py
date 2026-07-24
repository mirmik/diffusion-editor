"""Small adapter for the Big-LaMa TorchScript model."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image


DEFAULT_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/"
    "releases/download/v0.1.0/big-lama.pt"
)


def _padded_array(image: Image.Image, *, mode: str) -> np.ndarray:
    array = np.asarray(image.convert(mode))
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    else:
        array = np.transpose(array, (2, 0, 1))
    array = array.astype(np.float32) / 255.0
    _, height, width = array.shape
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    return np.pad(
        array,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="symmetric",
    )


class LamaModel:
    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_path = self._model_path()
        self._model = torch.jit.load(model_path, map_location=self._device)
        self._model.eval()
        self._model.to(self._device)

    def _model_path(self) -> Path:
        configured = os.environ.get("LAMA_MODEL")
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"LaMa TorchScript model not found: {path}"
                )
            return path

        url = os.environ.get("LAMA_MODEL_URL", DEFAULT_MODEL_URL)
        filename = Path(urlparse(url).path).name
        if not filename:
            raise RuntimeError(f"Invalid LAMA_MODEL_URL: {url}")
        checkpoint_dir = Path(self._torch.hub.get_dir()) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename
        if not path.is_file():
            self._torch.hub.download_url_to_file(url, str(path), progress=True)
        return path

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        torch = self._torch
        width, height = image.size
        image_array = _padded_array(image, mode="RGB")
        mask_array = _padded_array(mask, mode="L")
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(self._device)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(self._device)
        mask_tensor = (mask_tensor > 0).to(image_tensor.dtype)

        with torch.inference_mode():
            result = self._model(image_tensor, mask_tensor)

        result_array = (
            result[0, :, :height, :width]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array)
