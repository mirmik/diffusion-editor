"""GPUCompositor — FBO-based GPU layer compositing with premultiplied alpha."""

from __future__ import annotations

import logging

import numpy as np

from .layer_stack import LayerStack
from .layer import Layer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLSL sources
# ---------------------------------------------------------------------------

_VERT_SRC = """
#version 330 core
layout(location=0) in vec2 a_position;
layout(location=1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# Premultiplied alpha compositing fragment shader.
# Converts straight-alpha texture to premultiplied on the fly.
_COMPOSITE_FRAG_SRC = """
#version 330 core
uniform sampler2D u_texture;
uniform float u_opacity;
in vec2 v_uv;
out vec4 FragColor;
void main() {
    vec4 t = texture(u_texture, v_uv);
    float a = t.a * u_opacity;
    FragColor = vec4(t.rgb * a, a);
}
"""

# Un-premultiply shader: converts premultiplied alpha back to straight alpha
# for display by the Canvas base class.
_UNPREMUL_FRAG_SRC = """
#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 FragColor;
void main() {
    vec4 pm = texture(u_texture, v_uv);
    if (pm.a > 0.001)
        FragColor = vec4(pm.rgb / pm.a, pm.a);
    else
        FragColor = vec4(0.0);
}
"""


class GPUCompositor:
    """Composites a LayerStack on the GPU via OpenGL FBOs.

    Layers are uploaded as textures.  Compositing uses premultiplied-alpha
    blending in a main FBO, then un-premultiplied into a display FBO whose
    color attachment can be handed directly to the Canvas for rendering.
    """

    def __init__(self, layer_stack: LayerStack, graphics):
        self._stack = layer_stack
        self._graphics = graphics

        # Per-layer GPU textures, keyed by ``id(layer)``
        self._layer_textures: dict[int, object] = {}
        self._dirty_layers: set[int] = set()

        # FBOs (created lazily on first composite)
        self._main_fbo = None        # premultiplied-alpha accumulation
        self._display_fbo = None     # straight-alpha result
        self._fbo_w = 0
        self._fbo_h = 0

        # Temp FBO pool for group-opacity subtrees
        self._temp_fbos: list = []
        self._temp_fbos_in_use: int = 0

        # Shaders (created lazily)
        self._shader = None
        self._unpremul_shader = None

        self._dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def composite(self):
        """Composite all visible layers into the display FBO."""
        w, h = self._stack.width, self._stack.height
        if w == 0 or h == 0:
            return

        self._ensure_fbos(w, h)
        self._ensure_shaders()
        self._sync_dirty_textures()

        # --- Render into main FBO (premultiplied alpha) ---
        self._graphics.bind_framebuffer(self._main_fbo)
        self._graphics.set_viewport(0, 0, w, h)
        self._graphics.clear_color(0.0, 0.0, 0.0, 0.0)
        self._graphics.set_depth_test(False)
        self._graphics.set_blend(True)
        self._graphics.set_blend_func("one", "one_minus_src_alpha")

        self._shader.use()
        # Bottom-to-top: reversed list order (last in list = bottom)
        for layer in reversed(self._stack.layers):
            self._render_layer_tree(layer, self._main_fbo)

        # --- Un-premultiply pass into display FBO ---
        self._unpremultiply()

        self._graphics.bind_framebuffer(None)
        self._dirty = False

    def get_display_texture(self):
        """Return the GPU texture handle of the final composited image."""
        if self._display_fbo is None:
            return None
        return self._display_fbo.color_texture()

    def mark_dirty(self, layer: Layer | None = None):
        """Mark a layer (or the whole stack) for re-upload + re-composite."""
        self._dirty = True
        if layer is not None:
            self._dirty_layers.add(id(layer))
        else:
            # Mark all layers dirty
            for l in self._stack._all_layers_flat():
                self._dirty_layers.add(id(l))

    def rebuild(self):
        """Full rebuild after structural change (add/remove/reorder)."""
        self._cleanup_stale_textures()
        self._dirty = True
        self._dirty_layers = {id(l) for l in self._stack._all_layers_flat()}

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def readback(self) -> np.ndarray:
        """GPU -> CPU: read the display FBO into a numpy uint8 RGBA array."""
        if self._display_fbo is None:
            w, h = self._stack.width, self._stack.height
            if w == 0 or h == 0:
                return np.zeros((1, 1, 4), dtype=np.uint8)
            return np.zeros((h, w, 4), dtype=np.uint8)

        float_data = self._graphics.read_color_buffer_float(self._display_fbo)
        if float_data is None:
            logger.error("GPUCompositor.readback: read_color_buffer_float returned None")
            h, w = self._fbo_h, self._fbo_w
            return np.zeros((h, w, 4), dtype=np.uint8)

        result = np.clip(float_data * 255.0, 0, 255).astype(np.uint8)
        # OpenGL reads bottom-up; flip to top-down
        result = np.ascontiguousarray(result[::-1])
        return result

    def dispose(self):
        """Release all GPU resources."""
        for tex in self._layer_textures.values():
            tex.delete()
        self._layer_textures.clear()
        self._dirty_layers.clear()

        for fbo in self._temp_fbos:
            fbo.release()
        self._temp_fbos.clear()
        self._temp_fbos_in_use = 0

        if self._main_fbo is not None:
            self._main_fbo.release()
            self._main_fbo = None
        if self._display_fbo is not None:
            self._display_fbo.release()
            self._display_fbo = None

        if self._shader is not None:
            self._shader.release()
            self._shader = None
        if self._unpremul_shader is not None:
            self._unpremul_shader.release()
            self._unpremul_shader = None

    # ------------------------------------------------------------------
    # FBO / shader management
    # ------------------------------------------------------------------

    def _ensure_fbos(self, w: int, h: int):
        if (self._main_fbo is not None
                and self._fbo_w == w and self._fbo_h == h):
            return

        # Resize or create
        if self._main_fbo is not None:
            self._main_fbo.resize(w, h)
            self._display_fbo.resize(w, h)
            for fbo in self._temp_fbos:
                fbo.resize(w, h)
        else:
            self._main_fbo = self._graphics.create_framebuffer(w, h)
            self._display_fbo = self._graphics.create_framebuffer(w, h)

        self._fbo_w = w
        self._fbo_h = h

    def _ensure_shaders(self):
        if self._shader is not None:
            return
        self._shader = self._graphics.create_shader(
            _VERT_SRC, _COMPOSITE_FRAG_SRC)
        self._unpremul_shader = self._graphics.create_shader(
            _VERT_SRC, _UNPREMUL_FRAG_SRC)

    # ------------------------------------------------------------------
    # Texture sync
    # ------------------------------------------------------------------

    def _sync_dirty_textures(self):
        """Upload / update textures for layers whose content changed."""
        w, h = self._stack.width, self._stack.height
        for layer in self._stack._all_layers_flat():
            lid = id(layer)
            if lid not in self._layer_textures:
                tex = self._graphics.create_texture(
                    layer.image, w, h, channels=4, mipmap=False, clamp=True)
                self._layer_textures[lid] = tex
                self._dirty_layers.discard(lid)
            elif lid in self._dirty_layers:
                self._graphics.update_texture(
                    self._layer_textures[lid], layer.image, w, h, 4)
                self._dirty_layers.discard(lid)

    def _cleanup_stale_textures(self):
        """Remove textures for layers no longer in the stack."""
        live_ids = {id(l) for l in self._stack._all_layers_flat()}
        stale = [lid for lid in self._layer_textures if lid not in live_ids]
        for lid in stale:
            self._layer_textures[lid].delete()
            del self._layer_textures[lid]
            self._dirty_layers.discard(lid)

    # ------------------------------------------------------------------
    # Compositing helpers
    # ------------------------------------------------------------------

    def _render_layer_tree(self, layer: Layer, target_fbo):
        """Recursively composite a layer (and its children) into *target_fbo*."""
        if not layer.visible or layer.opacity <= 0:
            return

        has_children = bool(layer.children)
        needs_group_fbo = has_children and layer.opacity < 1.0

        if needs_group_fbo:
            temp = self._acquire_temp_fbo()
            self._graphics.bind_framebuffer(temp)
            self._graphics.set_viewport(0, 0, self._fbo_w, self._fbo_h)
            self._graphics.clear_color(0.0, 0.0, 0.0, 0.0)

            # Render children + this layer's own content at full opacity
            for child in reversed(layer.children):
                self._render_layer_tree(child, temp)
            self._draw_layer_quad(layer, opacity=1.0)

            # Blend temp result into the actual target at group opacity
            self._graphics.bind_framebuffer(target_fbo)
            self._graphics.set_viewport(0, 0, self._fbo_w, self._fbo_h)
            self._draw_texture_quad(temp.color_texture(), layer.opacity)
            self._release_temp_fbo(temp)
        else:
            if has_children:
                for child in reversed(layer.children):
                    self._render_layer_tree(child, target_fbo)
            self._draw_layer_quad(layer, layer.opacity)

    def _draw_layer_quad(self, layer: Layer, opacity: float):
        """Draw a layer's texture as a full-screen quad."""
        lid = id(layer)
        tex = self._layer_textures.get(lid)
        if tex is None:
            return
        self._draw_texture_quad(tex, opacity)

    def _draw_texture_quad(self, texture, opacity: float):
        """Draw a texture as a full-screen quad with the composite shader."""
        self._shader.use()
        texture.bind(0)
        self._shader.set_uniform_int("u_texture", 0)
        self._shader.set_uniform_float("u_opacity", opacity)
        self._graphics.draw_ui_textured_quad()

    def _unpremultiply(self):
        """Copy main FBO -> display FBO with un-premultiply shader."""
        self._graphics.bind_framebuffer(self._display_fbo)
        self._graphics.set_viewport(0, 0, self._fbo_w, self._fbo_h)
        self._graphics.clear_color(0.0, 0.0, 0.0, 0.0)
        self._graphics.set_blend(False)

        self._unpremul_shader.use()
        self._main_fbo.color_texture().bind(0)
        self._unpremul_shader.set_uniform_int("u_texture", 0)
        self._graphics.draw_ui_textured_quad()

        # Restore blending for next frame
        self._graphics.set_blend(True)
        self._graphics.set_blend_func("one", "one_minus_src_alpha")

    # ------------------------------------------------------------------
    # Temp FBO pool
    # ------------------------------------------------------------------

    def _acquire_temp_fbo(self):
        if self._temp_fbos_in_use < len(self._temp_fbos):
            fbo = self._temp_fbos[self._temp_fbos_in_use]
        else:
            fbo = self._graphics.create_framebuffer(self._fbo_w, self._fbo_h)
            self._temp_fbos.append(fbo)
        self._temp_fbos_in_use += 1
        return fbo

    def _release_temp_fbo(self, fbo):
        self._temp_fbos_in_use = max(0, self._temp_fbos_in_use - 1)
