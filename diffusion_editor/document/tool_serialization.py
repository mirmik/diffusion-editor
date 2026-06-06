"""Serialization and legacy migration helpers for layer tools."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import zipfile

import numpy as np
from PIL import Image

from .archive_serialization import load_pil_from_zip, save_array_to_zip
from .tool import DiffusionTool, InstructTool, LamaTool, Tool

Rect = tuple[int, int, int, int]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolLoadResult:
    tool: Tool | None
    legacy_patch_rect: Rect | None = None


def serialize_tool(tool: Tool, file_key: str) -> dict:
    if isinstance(tool, DiffusionTool):
        return _serialize_diffusion_tool(tool, file_key)
    if isinstance(tool, LamaTool):
        return _serialize_lama_tool(tool, file_key)
    if isinstance(tool, InstructTool):
        return _serialize_instruct_tool(tool, file_key)
    return {"tool_type": tool.tool_type}


def save_tool_assets(tool: Tool, zf: zipfile.ZipFile, file_key: str) -> None:
    if not isinstance(tool, (DiffusionTool, LamaTool, InstructTool)):
        return
    if tool.source_patch is not None:
        save_array_to_zip(
            zf,
            f"layers/{file_key}_source.npy",
            np.array(tool.source_patch),
        )


def load_tool(d: dict, zf: zipfile.ZipFile) -> ToolLoadResult:
    """Deserialize a tool and return migration metadata separately.

    Supports both new format (`tool_type`) and legacy layer-inline format
    (`type` = diffusion/lama/instruct).
    """
    tool_type = d.get("tool_type") or d.get("type")
    if tool_type == "diffusion":
        return _load_diffusion_tool(d, zf)
    if tool_type == "lama":
        return _load_lama_tool(d, zf)
    if tool_type == "instruct":
        return _load_instruct_tool(d, zf)
    if tool_type not in (None, "layer"):
        logger.warning("Unknown layer tool type while loading project: %s", tool_type)
    return ToolLoadResult(tool=None)


def _source_file(tool, file_key: str) -> str | None:
    if tool.source_patch is None:
        return None
    return f"layers/{file_key}_source.npy"


def _serialize_diffusion_tool(tool: DiffusionTool, file_key: str) -> dict:
    return {
        "tool_type": tool.tool_type,
        "mode": tool.mode,
        "source_file": _source_file(tool, file_key),
        "patch_x": tool.patch_x,
        "patch_y": tool.patch_y,
        "patch_w": tool.patch_w,
        "patch_h": tool.patch_h,
        "prompt": tool.prompt,
        "negative_prompt": tool.negative_prompt,
        "strength": tool.strength,
        "guidance_scale": tool.guidance_scale,
        "steps": tool.steps,
        "seed": tool.seed,
        "model_path": tool.model_path,
        "prediction_type": tool.prediction_type,
        "ip_adapter_layer_id": tool.ip_adapter_layer_id,
        "ip_adapter_layer_name_hint": tool.ip_adapter_layer_name_hint,
        "ip_adapter_scale": tool.ip_adapter_scale,
        "masked_content": tool.masked_content,
        "resize_to_model_resolution": tool.resize_to_model_resolution,
    }


def _serialize_lama_tool(tool: LamaTool, file_key: str) -> dict:
    return {
        "tool_type": tool.tool_type,
        "source_file": _source_file(tool, file_key),
        "patch_x": tool.patch_x,
        "patch_y": tool.patch_y,
        "patch_w": tool.patch_w,
        "patch_h": tool.patch_h,
    }


def _serialize_instruct_tool(tool: InstructTool, file_key: str) -> dict:
    return {
        "tool_type": tool.tool_type,
        "source_file": _source_file(tool, file_key),
        "patch_x": tool.patch_x,
        "patch_y": tool.patch_y,
        "patch_w": tool.patch_w,
        "patch_h": tool.patch_h,
        "instruction": tool.instruction,
        "image_guidance_scale": tool.image_guidance_scale,
        "guidance_scale": tool.guidance_scale,
        "steps": tool.steps,
        "seed": tool.seed,
    }


def _load_source_patch(d: dict, zf: zipfile.ZipFile) -> Image.Image | None:
    source_file = d.get("source_file")
    if source_file and source_file in zf.namelist():
        return load_pil_from_zip(zf, source_file, mode="RGB")
    return None


def _load_legacy_patch_rect(d: dict) -> Rect | None:
    rect = d.get("manual_patch_rect")
    return tuple(rect) if rect else None


def _load_diffusion_tool(d: dict, zf: zipfile.ZipFile) -> ToolLoadResult:
    tool = DiffusionTool(
        source_patch=_load_source_patch(d, zf),
        patch_x=d["patch_x"],
        patch_y=d["patch_y"],
        patch_w=d["patch_w"],
        patch_h=d["patch_h"],
        prompt=d["prompt"],
        negative_prompt=d["negative_prompt"],
        strength=d["strength"],
        guidance_scale=d["guidance_scale"],
        steps=d["steps"],
        seed=d["seed"],
        model_path=d.get("model_path", ""),
        prediction_type=d.get("prediction_type", ""),
        mode=d.get("mode", "inpaint"),
    )
    tool.ip_adapter_layer_id = d.get("ip_adapter_layer_id")
    tool.ip_adapter_layer_name_hint = d.get("ip_adapter_layer_name_hint", "")
    tool.ip_adapter_scale = d.get("ip_adapter_scale", 0.6)
    tool.masked_content = d.get("masked_content", "original")
    tool.resize_to_model_resolution = d.get("resize_to_model_resolution", False)
    return ToolLoadResult(
        tool=tool,
        legacy_patch_rect=_load_legacy_patch_rect(d),
    )


def _load_lama_tool(d: dict, zf: zipfile.ZipFile) -> ToolLoadResult:
    tool = LamaTool(
        source_patch=_load_source_patch(d, zf),
        patch_x=d["patch_x"],
        patch_y=d["patch_y"],
        patch_w=d["patch_w"],
        patch_h=d["patch_h"],
    )
    return ToolLoadResult(tool=tool)


def _load_instruct_tool(d: dict, zf: zipfile.ZipFile) -> ToolLoadResult:
    tool = InstructTool(
        source_patch=_load_source_patch(d, zf),
        patch_x=d["patch_x"],
        patch_y=d["patch_y"],
        patch_w=d["patch_w"],
        patch_h=d["patch_h"],
        instruction=d.get("instruction", ""),
        image_guidance_scale=d.get("image_guidance_scale", 1.5),
        guidance_scale=d.get("guidance_scale", 7.0),
        steps=d.get("steps", 20),
        seed=d.get("seed", -1),
    )
    return ToolLoadResult(
        tool=tool,
        legacy_patch_rect=_load_legacy_patch_rect(d),
    )
