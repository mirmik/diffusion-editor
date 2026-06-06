"""Agent tools for diffusion-editor — manipulate layers, canvas, etc."""

from __future__ import annotations

try:
    from nemor.core.tool_registry import ToolRegistry
except ImportError:  # pragma: no cover - depends on optional local install
    ToolRegistry = None

from ..document.commands import (
    AddLayerCommand,
    DrawGridCommand,
    DrawRectCommand,
    RemoveLayerCommand,
    SetLayerVisibilityCommand,
    SetLayerOpacityCommand,
)


def _get_layer_stack(config):
    return config["_layer_stack"]


def _get_document(config):
    return config["_document_service"]


def _find_layer(layer_stack, name: str):
    for layer in layer_stack._all_layers_flat():
        if layer.name == name:
            return layer
    return None


def _on_main(config, func):
    run = config.get("_run_on_main_thread")
    if run is None:
        return func()
    return run(func)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ndc_to_pixels(ndc_x, ndc_y, ndc_w, ndc_h, cw, ch):
    """Convert NDC rect (0–1) to pixel rect."""
    px = int(ndc_x * cw)
    py = int(ndc_y * ch)
    pw = int(ndc_w * cw)
    ph = int(ndc_h * ch)
    return px, py, pw, ph


def _save_image_for_llm(img, config):
    import hashlib
    import io
    import os

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_data = buf.getvalue()

    try:
        from nemor.images import IMAGE_EXT_BY_MIME, prepare_image_for_llm
        data, mime = prepare_image_for_llm(png_data, "image/png")
        ext = IMAGE_EXT_BY_MIME[mime]
    except Exception:
        data, mime = png_data, "image/png"
        ext = ".png"

    nemor_dir = config.get("nemor_dir") or os.path.expanduser("~/.nemor")
    media_dir = os.path.join(nemor_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    file_hash = hashlib.sha1(data).hexdigest()[:12]
    dest_name = f"canvas_{file_hash}{ext}"
    dest_path = os.path.join(media_dir, dest_name)
    with open(dest_path, "wb") as f:
        f.write(data)

    w, h = img.size
    size = len(data)
    size_str = f"{size / 1024:.0f}KB" if size >= 1024 else f"{size}B"
    filename = f"canvas_{w}x{h}{ext}"
    text = f"Canvas: {w}x{h} ({size_str})"
    return {
        "text": text,
        "images": [{
            "path": dest_path,
            "url": f"/media/{dest_name}",
            "filename": filename,
            "mime": mime,
            "size": size,
        }],
    }


# ── Tool implementations ────────────────────────────────────────────────────


def _tool_list_layers(args, config):
    def op():
        layer_stack = _get_layer_stack(config)
        layers = layer_stack._all_layers_flat()
        if not layers:
            return "(no layers)"
        lines = []
        for i, layer in enumerate(layers):
            active = " [active]" if layer is layer_stack.active_layer else ""
            lines.append(
                f"  [{i}] {layer.name}: type={type(layer).__name__}, "
                f"visible={layer.visible}, opacity={layer.opacity:.0%}{active}"
            )
        w, h = layer_stack.width, layer_stack.height
        lines.append(f"Canvas: {w}x{h}, {len(layers)} layers total")
        return "\n".join(lines)

    return _on_main(config, op)


def _tool_add_layer(args, config):
    name = args["name"]

    def op():
        document = _get_document(config)
        document.execute(AddLayerCommand(name=name))
        return f"Layer '{name}' created."

    return _on_main(config, op)


def _tool_remove_layer(args, config):
    name = args["name"]

    def op():
        document = _get_document(config)
        layer_stack = _get_layer_stack(config)
        layer = _find_layer(layer_stack, name)
        if layer is None:
            return f"Layer '{name}' not found."
        document.execute(RemoveLayerCommand(layer=layer))
        return f"Layer '{name}' removed."

    return _on_main(config, op)


def _tool_set_layer_visibility(args, config):
    name = args["name"]
    visible = args["visible"]

    def op():
        document = _get_document(config)
        layer_stack = _get_layer_stack(config)
        layer = _find_layer(layer_stack, name)
        if layer is None:
            return f"Layer '{name}' not found."
        document.execute(SetLayerVisibilityCommand(layer=layer, visible=visible))
        return f"Layer '{name}' visibility set to {visible}."

    return _on_main(config, op)


def _tool_set_layer_opacity(args, config):
    name = args["name"]
    opacity = float(args["opacity"])
    opacity = max(0.0, min(1.0, opacity))

    def op():
        document = _get_document(config)
        layer_stack = _get_layer_stack(config)
        layer = _find_layer(layer_stack, name)
        if layer is None:
            return f"Layer '{name}' not found."
        document.execute(SetLayerOpacityCommand(layer=layer, opacity=opacity))
        return f"Layer '{name}' opacity set to {opacity:.0%}."

    return _on_main(config, op)


def _tool_get_canvas_info(args, config):
    def op():
        layer_stack = _get_layer_stack(config)
        w, h = layer_stack.width, layer_stack.height
        active = layer_stack.active_layer
        active_name = active.name if active else "(none)"
        layers = layer_stack._all_layers_flat()
        return (
            f"Canvas: {w}x{h}\n"
            f"Active layer: {active_name}\n"
            f"Layer count: {len(layers)}"
        )

    return _on_main(config, op)


def _tool_draw_rect(args, config):
    def op():
        document = _get_document(config)
        layer_stack = _get_layer_stack(config)
        active = layer_stack.active_layer
        if active is None:
            return "No active layer. Select or create a layer first."

        ndc_x = float(args["x"])
        ndc_y = float(args["y"])
        ndc_w = float(args["width"])
        ndc_h = float(args["height"])
        px, py, pw, ph = _ndc_to_pixels(
            ndc_x, ndc_y, ndc_w, ndc_h,
            layer_stack.width, layer_stack.height,
        )

        r = int(args.get("r", 255))
        g = int(args.get("g", 0))
        b = int(args.get("b", 0))
        a = int(args.get("a", 255))
        color = (r, g, b, a)
        thickness = int(args.get("thickness", 2))

        cmd = DrawRectCommand(
            layer=active,
            x=px, y=py, width=pw, height=ph,
            color=color,
            thickness=thickness,
        )
        document.execute(cmd)
        return (
            f"Rectangle drawn on '{active.name}': "
            f"ndc=({ndc_x:.3f}, {ndc_y:.3f}) {ndc_w:.3f}x{ndc_h:.3f} "
            f"-> px=({px}, {py}) {pw}x{ph}, "
            f"color=rgba({r},{g},{b},{a}), thickness={thickness}"
        )

    return _on_main(config, op)


def _tool_draw_grid(args, config):
    def op():
        document = _get_document(config)
        layer_stack = _get_layer_stack(config)
        active = layer_stack.active_layer
        if active is None:
            return "No active layer. Select or create a layer first."

        sx = int(args["sections_x"])
        sy = int(args["sections_y"])
        r = int(args.get("r", 255))
        g = int(args.get("g", 0))
        b = int(args.get("b", 0))
        a = int(args.get("a", 128))
        color = (r, g, b, a)
        thickness = int(args.get("thickness", 1))

        cmd = DrawGridCommand(
            layer=active,
            sections_x=sx,
            sections_y=sy,
            color=color,
            thickness=thickness,
        )
        document.execute(cmd)
        return (
            f"Grid drawn on '{active.name}': "
            f"{sx}x{sy} sections ({sx + 1}×{sy + 1} lines), "
            f"color=rgba({r},{g},{b},{a}), thickness={thickness}"
        )

    return _on_main(config, op)


def _tool_view_canvas(args, config):
    def op():
        from PIL import Image

        layer_stack = _get_layer_stack(config)
        arr = layer_stack.composite()
        img = Image.fromarray(arr, "RGBA")
        result = _save_image_for_llm(img, config)
        layer_count = len(layer_stack._all_layers_flat())
        result["text"] = f"Canvas: {img.size[0]}x{img.size[1]}, {layer_count} layers"
        return result

    return _on_main(config, op)


def _tool_view_canvas_region(args, config):
    def op():
        from PIL import Image

        layer_stack = _get_layer_stack(config)
        arr = layer_stack.composite()
        full_img = Image.fromarray(arr, "RGBA")
        cw, ch = full_img.size

        ndc_x = float(args["x"])
        ndc_y = float(args["y"])
        ndc_w = float(args["width"])
        ndc_h = float(args["height"])
        px, py, pw, ph = _ndc_to_pixels(ndc_x, ndc_y, ndc_w, ndc_h, cw, ch)

        # Clamp to image bounds
        x0 = max(0, px)
        y0 = max(0, py)
        x1 = min(cw, px + pw)
        y1 = min(ch, py + ph)
        if x0 >= x1 or y0 >= y1:
            return "Region is outside the canvas."

        crop = full_img.crop((x0, y0, x1, y1))
        result = _save_image_for_llm(crop, config)
        result["text"] = (
            f"Region: ndc=({ndc_x:.3f}, {ndc_y:.3f}) {ndc_w:.3f}x{ndc_h:.3f} "
            f"-> px=({x0}, {y0}) {x1 - x0}x{y1 - y0}"
        )
        return result

    return _on_main(config, op)


# ── Registry factory ────────────────────────────────────────────────────────


def create_editor_tool_registry() -> ToolRegistry | None:
    if ToolRegistry is None:
        return None

    registry = ToolRegistry()

    registry.register("list_layers", _tool_list_layers, {
        "type": "function",
        "function": {
            "name": "list_layers",
            "description": "List all layers in the document with their names, types, visibility, opacity, and active status. Includes canvas dimensions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    })

    registry.register("add_layer", _tool_add_layer, {
        "type": "function",
        "function": {
            "name": "add_layer",
            "description": "Add a new empty layer to the document. The layer appears at the top of the stack.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Layer name"},
                },
                "required": ["name"],
            },
        },
    })

    registry.register("remove_layer", _tool_remove_layer, {
        "type": "function",
        "function": {
            "name": "remove_layer",
            "description": "Remove a layer by name. This action is undoable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact layer name to remove"},
                },
                "required": ["name"],
            },
        },
    })

    registry.register("set_layer_visibility", _tool_set_layer_visibility, {
        "type": "function",
        "function": {
            "name": "set_layer_visibility",
            "description": "Show or hide a layer by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact layer name"},
                    "visible": {"type": "boolean", "description": "True = visible, False = hidden"},
                },
                "required": ["name", "visible"],
            },
        },
    })

    registry.register("set_layer_opacity", _tool_set_layer_opacity, {
        "type": "function",
        "function": {
            "name": "set_layer_opacity",
            "description": "Set layer opacity (0.0 = fully transparent, 1.0 = fully opaque).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact layer name"},
                    "opacity": {"type": "number", "description": "Opacity value from 0.0 to 1.0"},
                },
                "required": ["name", "opacity"],
            },
        },
    })

    registry.register("get_canvas_info", _tool_get_canvas_info, {
        "type": "function",
        "function": {
            "name": "get_canvas_info",
            "description": "Get canvas dimensions, active layer name, and total layer count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    })

    registry.register("view_canvas", _tool_view_canvas, {
        "type": "function",
        "function": {
            "name": "view_canvas",
            "description": "Capture the full canvas as an image so you can see the composite result of all visible layers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    })

    registry.register("view_canvas_region", _tool_view_canvas_region, {
        "type": "function",
        "function": {
            "name": "view_canvas_region",
            "description": "Capture a rectangular region of the canvas in NDC coordinates (0,0 = top-left, 1,1 = bottom-right). Use this to zoom in on details or inspect specific areas at full resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Left edge X in NDC (0.0 to 1.0)"},
                    "y": {"type": "number", "description": "Top edge Y in NDC (0.0 to 1.0)"},
                    "width": {"type": "number", "description": "Width in NDC (0.0 to 1.0)"},
                    "height": {"type": "number", "description": "Height in NDC (0.0 to 1.0)"},
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    })

    registry.register("draw_rect", _tool_draw_rect, {
        "type": "function",
        "function": {
            "name": "draw_rect",
            "description": "Draw a rectangle outline on the active layer using NDC (normalized device coordinates) where (0,0) is top-left and (1,1) is bottom-right. Use view_canvas first to see the image, then specify the rectangle in NDC space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Left edge X in NDC (0.0 to 1.0)"},
                    "y": {"type": "number", "description": "Top edge Y in NDC (0.0 to 1.0)"},
                    "width": {"type": "number", "description": "Width in NDC (0.0 to 1.0)"},
                    "height": {"type": "number", "description": "Height in NDC (0.0 to 1.0)"},
                    "r": {"type": "integer", "description": "Red channel (0-255). Default 255."},
                    "g": {"type": "integer", "description": "Green channel (0-255). Default 0."},
                    "b": {"type": "integer", "description": "Blue channel (0-255). Default 0."},
                    "a": {"type": "integer", "description": "Alpha channel (0-255). Default 255."},
                    "thickness": {"type": "integer", "description": "Line thickness in pixels. Default 2."},
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    })

    registry.register("draw_grid", _tool_draw_grid, {
        "type": "function",
        "function": {
            "name": "draw_grid",
            "description": "Draw a grid overlay on the active layer to help with spatial orientation. Divides the canvas into equal sections, making it easier to estimate NDC coordinates for objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections_x": {"type": "integer", "description": "Number of vertical divisions (columns)."},
                    "sections_y": {"type": "integer", "description": "Number of horizontal divisions (rows)."},
                    "r": {"type": "integer", "description": "Red channel (0-255). Default 255."},
                    "g": {"type": "integer", "description": "Green channel (0-255). Default 0."},
                    "b": {"type": "integer", "description": "Blue channel (0-255). Default 0."},
                    "a": {"type": "integer", "description": "Alpha channel (0-255). Default 128 (semi-transparent)."},
                    "thickness": {"type": "integer", "description": "Line thickness in pixels. Default 1."},
                },
                "required": ["sections_x", "sections_y"],
            },
        },
    })

    return registry
