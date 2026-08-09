"""GPUCompositor — tgfx2 native layer compositing with premultiplied alpha."""

from __future__ import annotations

import struct

import numpy as np
from tcbase import log
from termin.geombase import LinearColor
from tgfx import TextureEncoding

from ..color import linear_rgba_to_srgb8
from ..document.layer_stack import LayerStack
from ..document.layer import Layer

from tgfx._tgfx_native import (
    CULL_NONE,
    ShaderArtifactPolicy,
    ShaderLanguage,
    TcShader,
    Tgfx2Context,
    Tgfx2PixelFormat,
    Tgfx2TextureHandle,
    Tgfx2BlendFactor,
    tc_shader_ensure_tgfx2,
)

Rect = tuple[int, int, int, int]
_TRANSPARENT_LINEAR = LinearColor(0.0, 0.0, 0.0, 0.0)
_LAYER_TEXTURE_ENCODING = TextureEncoding.SRGB

# ---------------------------------------------------------------------------
# Scoped Slang sources
# ---------------------------------------------------------------------------
#
# The Termin compiler produces backend artifacts and resource-layout metadata
# from these sources. Explicit TerminScope attributes keep transient texture
# and per-draw uniform bindings portable across OpenGL and Vulkan.

_VERT_SRC = """import termin_prelude;

struct VertexInput {
    float3 position : POSITION;
    float4 uv_pad : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = termin_to_native_clip(float4(input.position.xy, 0.0, 1.0));
    output.uv = input.uv_pad.xy;
    return output;
}
"""

_COMPOSITE_FRAG_SRC = """import termin_prelude;

struct CompositeParams {
    float opacity;
    float source_premultiplied;
};

[[TerminScope("draw")]]
ConstantBuffer<CompositeParams> u_composite;

[[TerminScope("transient")]]
Sampler2D u_texture;

struct FragmentInput {
    float4 screen_position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fs_main(FragmentInput input) {
    float4 texel = u_texture.Sample(input.uv);
    float alpha = texel.a * u_composite.opacity;
    float3 rgb = texel.rgb * u_composite.opacity;
    if (u_composite.source_premultiplied < 0.5)
        rgb *= texel.a;
    FragmentOutput output;
    output.color = float4(rgb, alpha);
    return output;
}
"""

_UNPREMUL_FRAG_SRC = """import termin_prelude;

[[TerminScope("transient")]]
Sampler2D u_texture;

struct FragmentInput {
    float4 screen_position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fs_main(FragmentInput input) {
    float4 premultiplied = u_texture.Sample(input.uv);
    FragmentOutput output;
    if (premultiplied.a > 0.001) {
        output.color = float4(premultiplied.rgb / premultiplied.a,
                              premultiplied.a);
    } else {
        output.color = float4(0.0);
    }
    return output;
}
"""

# Python-side std140 layout of CompositeParams. A one-float constant buffer
# still occupies a 16-byte block on all supported backends.
_COMPOSITE_PARAMS_FMT = "=ff8x"  # opacity + source kind + std140 padding


def _full_screen_quad_verts() -> np.ndarray:
    # Vertex format matches draw_immediate_triangles: 7 floats per vertex
    # (x, y, z, r, g, b, a). We repurpose RGBA as (u, v, _, _) so the
    # shader can read UV out of loc 1 — same pattern tgfx2 bindings
    # document for immediate quads.
    # Clip-space -1..1 covers the whole offscreen target.
    return np.array([
        # x,    y,    z,    u,    v,    _,    _
        -1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0, 0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0, 0.0, 0.0,
        -1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0, 0.0, 0.0,
        -1.0,  1.0, 0.0,  0.0, 1.0, 0.0, 0.0,
    ], dtype=np.float32)


def _canvas_quad_verts(x0: int, y0: int, x1: int, y1: int,
                       canvas_w: int, canvas_h: int) -> np.ndarray:
    left = (float(x0) / canvas_w) * 2.0 - 1.0
    right = (float(x1) / canvas_w) * 2.0 - 1.0
    bottom = (float(y0) / canvas_h) * 2.0 - 1.0
    top = (float(y1) / canvas_h) * 2.0 - 1.0
    return np.array([
        left,  bottom, 0.0,  0.0, 0.0, 0.0, 0.0,
        right, bottom, 0.0,  1.0, 0.0, 0.0, 0.0,
        right, top,    0.0,  1.0, 1.0, 0.0, 0.0,
        left,  bottom, 0.0,  0.0, 0.0, 0.0, 0.0,
        right, top,    0.0,  1.0, 1.0, 0.0, 0.0,
        left,  top,    0.0,  0.0, 1.0, 0.0, 0.0,
    ], dtype=np.float32)


class GPUCompositor:
    """Composites a LayerStack on the GPU using tgfx2 native textures.

    Layers upload as sRGB ``Tgfx2TextureHandle`` RGBA8 textures so sampling
    decodes RGB to linear light while leaving alpha linear. The first pass
    composites into a premultiplied linear RGBA16F target; a second pass
    un-premultiplies into a straight linear RGBA16F display texture whose
    handle is handed to the native Canvas.
    """

    def __init__(self, layer_stack: LayerStack,
                 graphics: Tgfx2Context):
        """
        Parameters
        ----------
        layer_stack : LayerStack
            Model to composite.
        graphics : Tgfx2Context
            The process-wide tgfx2 context (device + RenderContext2).
            Required — GPUCompositor never spawns its own Tgfx2Context,
            because that would create a second IRenderDevice and break
            cross-renderer TextureHandle sharing. Obtain from the host
            via ``Tgfx2Context.from_runtime(...)``.
        """
        if graphics is None:
            raise ValueError(
                "GPUCompositor requires a graphics= Tgfx2Context. Get one "
                "from the host (Tgfx2Context.from_runtime).")
        self._stack = layer_stack

        # Per-layer GPU textures, keyed by ``id(layer)`` — Tgfx2TextureHandle.
        self._layer_textures: dict[int, Tgfx2TextureHandle] = {}
        self._layer_tex_size: dict[int, tuple[int, int]] = {}
        self._layer_tex_revision: dict[int, int] = {}
        self._dirty_layers: set[int] = set()
        self._dirty_layer_regions: dict[int, Rect] = {}

        # Offscreen color attachments (created lazily on first composite).
        self._main_tex: Tgfx2TextureHandle | None = None
        self._display_tex: Tgfx2TextureHandle | None = None
        self._fbo_w = 0
        self._fbo_h = 0

        # Temp color attachments pool for group-opacity subtrees.
        self._temp_texs: list[Tgfx2TextureHandle] = []
        self._temp_texs_in_use: int = 0

        # tgfx2 context + compiled shaders (shaders compiled lazily on
        # first composite).
        self._graphics: Tgfx2Context = graphics
        self._ctx = None
        self._composite_vs = None
        self._composite_fs = None
        self._composite_shader = None
        self._unpremul_vs = None
        self._unpremul_fs = None
        self._unpremul_shader = None

        self._quad_verts = _full_screen_quad_verts()

        self._dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def composite(self):
        """Composite all visible layers into the display texture."""
        w, h = self._stack.width, self._stack.height
        if w == 0 or h == 0:
            return

        self._ensure_context()
        self._ensure_attachments(w, h)
        self._sync_dirty_textures()

        ctx = self._ctx
        opened_frame = not ctx.in_frame
        if opened_frame:
            ctx.begin_frame()
        try:
            # --- Main pass: accumulate layers with premultiplied-alpha blending ---
            ctx.begin_pass(
                color=self._main_tex,
                depth=None,
                clear_linear_color=_TRANSPARENT_LINEAR,
                clear_depth=1.0,
                clear_depth_enabled=False,
            )
            ctx.set_viewport(0, 0, w, h)
            ctx.clear_scissor()
            ctx.set_cull(CULL_NONE)
            ctx.set_depth_test(False)
            ctx.set_depth_write(False)
            ctx.set_color_mask(True, True, True, True)
            ctx.set_blend(True)
            ctx.set_blend_func(Tgfx2BlendFactor.One,
                               Tgfx2BlendFactor.OneMinusSrcAlpha)
            ctx.bind_shader(self._composite_vs, self._composite_fs)
            ctx.use_shader_resource_layout(self._composite_shader)

            # Bottom-to-top: reversed list order (last in list = bottom).
            # Need to close the current pass to re-target for nested
            # group-opacity subtrees, so we track the active target stack.
            for layer in reversed(self._stack.layers):
                self._render_layer_tree(layer, self._main_tex)

            ctx.end_pass()

            # --- Un-premultiply pass: main_tex → display_tex ---
            ctx.begin_pass(
                color=self._display_tex,
                depth=None,
                clear_linear_color=_TRANSPARENT_LINEAR,
                clear_depth=1.0,
                clear_depth_enabled=False,
            )
            ctx.set_viewport(0, 0, w, h)
            ctx.clear_scissor()
            ctx.set_cull(CULL_NONE)
            ctx.set_depth_test(False)
            ctx.set_depth_write(False)
            ctx.set_color_mask(True, True, True, True)
            ctx.set_blend(False)
            ctx.bind_shader(self._unpremul_vs, self._unpremul_fs)
            ctx.use_shader_resource_layout(self._unpremul_shader)
            ctx.bind_texture_by_name("u_texture", self._main_tex)
            ctx.draw_immediate_triangles(self._quad_verts, 6)
        finally:
            # An exception must never strand a borrowed RenderContext2 in an
            # open pass/frame. UIRenderer deliberately does not close frames
            # opened by another renderer; leaving ours alive makes Vulkan's
            # transient buffers and descriptor pool grow until exhaustion.
            try:
                ctx.end_pass()
            finally:
                if opened_frame:
                    ctx.end_frame()
        self._dirty = False

    @property
    def display_tex(self):
        """tgfx2 TextureHandle of the composited image. Valid in the
        renderer's device because the compositor was constructed with
        a borrowed context.
        """
        return self._display_tex

    def display_size(self) -> tuple[int, int]:
        """(width, height) of the current display texture, or (0, 0)."""
        return (self._fbo_w, self._fbo_h)

    def mark_dirty(
            self,
            layer: Layer | None = None,
            rect: Rect | None = None) -> None:
        """Mark layer pixels for upload and request a new composite.

        ``rect`` uses layer-local texture coordinates. A full dirty mark
        dominates any partial region accumulated for the same layer.
        """
        self._dirty = True
        if layer is None:
            for l in self._stack._all_layers_flat():
                lid = id(l)
                self._dirty_layers.add(lid)
                self._dirty_layer_regions.pop(lid, None)
            return

        lid = id(layer)
        if rect is None:
            self._dirty_layers.add(lid)
            self._dirty_layer_regions.pop(lid, None)
            return
        if lid in self._dirty_layers:
            return

        x0, y0, x1, y1 = rect
        clipped = (
            max(0, x0),
            max(0, y0),
            min(layer.width, x1),
            min(layer.height, y1),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return
        previous = self._dirty_layer_regions.get(lid)
        self._dirty_layer_regions[lid] = (
            clipped if previous is None else self._union_rect(previous, clipped)
        )

    def mark_composite_dirty(self) -> None:
        """Request recomposition without re-uploading unchanged layer pixels."""
        self._dirty = True

    def rebuild(self):
        """Recompose after model changes; revisions decide pixel uploads."""
        self._cleanup_stale_textures()
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def readback(self) -> np.ndarray:
        """GPU -> CPU: read the display texture into a numpy uint8 RGBA array."""
        if self._display_tex is None or self._graphics is None:
            w, h = self._stack.width, self._stack.height
            if w == 0 or h == 0:
                return np.zeros((1, 1, 4), dtype=np.uint8)
            return np.zeros((h, w, 4), dtype=np.uint8)

        result = self._readback_texture(
            self._display_tex,
            self._fbo_w,
            self._fbo_h,
            "display",
            encode_srgb=True,
        )
        if result is not None:
            return result
        return np.zeros((self._fbo_h, self._fbo_w, 4), dtype=np.uint8)

    def diagnostic_readbacks(
            self, layer: Layer | None = None) -> dict[str, np.ndarray | None]:
        """Read the upload, premultiplied and display stages for smoke errors."""
        source = None
        if layer is not None:
            texture = self._layer_textures.get(id(layer))
            if texture is not None:
                source = self._readback_texture(
                    texture,
                    layer.width,
                    layer.height,
                    "source upload",
                    encode_srgb=True,
                )
        return {
            "source upload": source,
            "premultiplied compositor": self._readback_texture(
                self._main_tex, self._fbo_w, self._fbo_h, "main"),
            "display compositor": self._readback_texture(
                self._display_tex,
                self._fbo_w,
                self._fbo_h,
                "display",
                encode_srgb=True,
            ),
        }

    def dispose(self):
        """Release all GPU resources."""
        if self._graphics is None:
            # Nothing was ever created.
            self._layer_textures.clear()
            self._layer_tex_size.clear()
            self._layer_tex_revision.clear()
            self._dirty_layers.clear()
            self._dirty_layer_regions.clear()
            self._temp_texs.clear()
            return

        for tex in self._layer_textures.values():
            self._graphics.destroy_texture(tex)
        self._layer_textures.clear()
        self._layer_tex_size.clear()
        self._layer_tex_revision.clear()
        self._dirty_layers.clear()
        self._dirty_layer_regions.clear()

        for tex in self._temp_texs:
            self._graphics.destroy_texture(tex)
        self._temp_texs.clear()
        self._temp_texs_in_use = 0

        if self._main_tex is not None:
            self._graphics.destroy_texture(self._main_tex)
            self._main_tex = None
        if self._display_tex is not None:
            self._graphics.destroy_texture(self._display_tex)
            self._display_tex = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_context(self):
        if self._ctx is None:
            self._ctx = self._graphics.context
        if self._composite_vs is None:
            self._composite_shader, composite = self._ensure_shader(
                "diffusion-editor-gpu-composite", _COMPOSITE_FRAG_SRC)
            self._composite_vs, self._composite_fs = composite.vs, composite.fs
            self._unpremul_shader, unpremul = self._ensure_shader(
                "diffusion-editor-gpu-unpremultiply", _UNPREMUL_FRAG_SRC)
            self._unpremul_vs, self._unpremul_fs = unpremul.vs, unpremul.fs
            if not (self._composite_vs and self._composite_fs
                    and self._unpremul_vs and self._unpremul_fs):
                raise RuntimeError("GPUCompositor: shader compile failed")

    def _ensure_shader(self, uuid: str, fragment_source: str):
        """Build a stable TcShader with the current symbolic resource layout."""
        shader = TcShader.get_or_create(uuid)
        # The current Termin shader contract validates source text against an
        # already selected language/artifact policy.
        shader.set_language(ShaderLanguage.SLANG)
        shader.set_artifact_policy(ShaderArtifactPolicy.REQUIRED)
        shader.set_sources_with_entries(
            _VERT_SRC,
            fragment_source,
            "",
            uuid,
            f"diffusion_editor/canvas/{uuid}.slang",
            "vs_main",
            "fs_main",
        )
        pair = tc_shader_ensure_tgfx2(self._ctx, shader)
        return shader, pair

    def _readback_texture(
            self,
            texture,
            width: int,
            height: int,
            label: str,
            *,
            encode_srgb: bool = False) -> np.ndarray | None:
        if (
                texture is None
                or self._graphics is None
                or width <= 0
                or height <= 0):
            return None
        # Use the Tgfx2Context's own device. Pulling RenderingManager into
        # this standalone editor would load the full Termin runtime.
        device = self._graphics.device
        if device is None:
            log.error(f"GPUCompositor {label} readback: device unavailable")
            return None
        buf = np.empty((height * width * 4,), dtype=np.float32)
        if not device.read_texture_rgba_float(texture, buf):
            log.error(f"GPUCompositor {label} readback failed")
            return None
        float_data = buf.reshape((height, width, 4))
        if encode_srgb:
            result = linear_rgba_to_srgb8(float_data)
        else:
            result = np.rint(
                np.clip(float_data, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        # IRenderDevice normalizes texture readback to top-down rows on every
        # backend; applying an OpenGL-specific flip here mirrored CPU users.
        return np.ascontiguousarray(result)

    def _ensure_attachments(self, w: int, h: int):
        if (self._main_tex is not None
                and self._fbo_w == w and self._fbo_h == h):
            return

        # Destroy old and recreate — the sizes change rarely (canvas resize).
        if self._main_tex is not None:
            self._graphics.destroy_texture(self._main_tex)
        if self._display_tex is not None:
            self._graphics.destroy_texture(self._display_tex)
        for tex in self._temp_texs:
            self._graphics.destroy_texture(tex)
        self._temp_texs.clear()
        self._temp_texs_in_use = 0

        # All compositor targets hold linear-light premultiplied or straight
        # data. Float16 avoids repeated quantization and preserves dark tones.
        self._main_tex = self._graphics.create_color_attachment(
            w, h, Tgfx2PixelFormat.RGBA16F)
        self._display_tex = self._graphics.create_color_attachment(
            w, h, Tgfx2PixelFormat.RGBA16F)
        self._fbo_w = w
        self._fbo_h = h

    # ------------------------------------------------------------------
    # Texture sync
    # ------------------------------------------------------------------

    def _sync_dirty_textures(self):
        for layer in self._stack._all_layers_flat():
            lid = id(layer)
            w, h = layer.width, layer.height
            revision = int(getattr(layer, "pixel_revision", 0))
            if lid not in self._layer_textures:
                img = np.ascontiguousarray(layer.image).reshape(-1)
                tex = self._graphics.create_texture_rgba8(
                    w, h, img, _LAYER_TEXTURE_ENCODING)
                self._layer_textures[lid] = tex
                self._layer_tex_size[lid] = (w, h)
                self._layer_tex_revision[lid] = revision
                self._clear_layer_dirty(lid)
                continue

            prev_w, prev_h = self._layer_tex_size.get(lid, (0, 0))
            if prev_w != w or prev_h != h:
                img = np.ascontiguousarray(layer.image).reshape(-1)
                self._graphics.destroy_texture(self._layer_textures[lid])
                tex = self._graphics.create_texture_rgba8(
                    w, h, img, _LAYER_TEXTURE_ENCODING)
                self._layer_textures[lid] = tex
                self._layer_tex_size[lid] = (w, h)
                self._layer_tex_revision[lid] = revision
                self._clear_layer_dirty(lid)
                continue

            if (
                    lid in self._dirty_layers
                    or (
                        lid not in self._dirty_layer_regions
                        and self._layer_tex_revision.get(lid) != revision
                    )):
                img = np.ascontiguousarray(layer.image).reshape(-1)
                self._graphics.upload_texture(self._layer_textures[lid], img)
                self._layer_tex_revision[lid] = revision
                self._clear_layer_dirty(lid)
                continue

            rect = self._dirty_layer_regions.pop(lid, None)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            region = np.ascontiguousarray(
                layer.image[y0:y1, x0:x1],
            ).reshape(-1)
            self._graphics.upload_texture_region(
                self._layer_textures[lid],
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                region,
            )
            self._layer_tex_revision[lid] = revision

    def _clear_layer_dirty(self, lid: int) -> None:
        self._dirty_layers.discard(lid)
        self._dirty_layer_regions.pop(lid, None)

    @staticmethod
    def _union_rect(first: Rect, second: Rect) -> Rect:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    def _cleanup_stale_textures(self):
        live_ids = {id(l) for l in self._stack._all_layers_flat()}
        stale = [lid for lid in self._layer_textures if lid not in live_ids]
        if not stale:
            return
        for lid in stale:
            if self._graphics is not None:
                self._graphics.destroy_texture(self._layer_textures[lid])
            del self._layer_textures[lid]
            self._layer_tex_size.pop(lid, None)
            self._layer_tex_revision.pop(lid, None)
            self._clear_layer_dirty(lid)

    # ------------------------------------------------------------------
    # Compositing helpers
    # ------------------------------------------------------------------

    def _render_layer_tree(self, layer: Layer, target_tex: Tgfx2TextureHandle):
        """Recursively composite a layer (and its children) into *target_tex*."""
        if not self._stack.is_layer_tree_visible_for_composition(layer):
            return

        own_visible = self._stack.is_layer_visible_for_composition(layer)
        if own_visible and layer.opacity <= 0:
            return
        has_children = bool(layer.children)
        needs_group = own_visible and has_children and layer.opacity < 1.0

        if needs_group:
            # Open a nested pass on a temp attachment.
            temp = self._acquire_temp_tex()
            ctx = self._ctx
            # End the current (parent) pass, render the group into temp,
            # then reopen the parent pass on target_tex to blend temp in.
            ctx.end_pass()
            ctx.begin_pass(
                color=temp,
                depth=None,
                clear_linear_color=_TRANSPARENT_LINEAR,
                clear_depth=1.0,
                clear_depth_enabled=False,
            )
            ctx.set_viewport(0, 0, self._fbo_w, self._fbo_h)
            ctx.clear_scissor()
            ctx.set_cull(CULL_NONE)
            ctx.set_depth_test(False)
            ctx.set_depth_write(False)
            ctx.set_color_mask(True, True, True, True)
            ctx.set_blend(True)
            ctx.set_blend_func(Tgfx2BlendFactor.One,
                               Tgfx2BlendFactor.OneMinusSrcAlpha)
            ctx.bind_shader(self._composite_vs, self._composite_fs)
            ctx.use_shader_resource_layout(self._composite_shader)

            for child in reversed(layer.children):
                self._render_layer_tree(child, temp)
            self._draw_layer_quad(layer, opacity=1.0)

            ctx.end_pass()
            # Reopen the parent pass without a clear color to preserve the
            # existing contents of target_tex.
            ctx.begin_pass(
                color=target_tex,
                depth=None,
                clear_linear_color=None,
                clear_depth=1.0,
                clear_depth_enabled=False,
            )
            ctx.set_viewport(0, 0, self._fbo_w, self._fbo_h)
            ctx.clear_scissor()
            ctx.set_cull(CULL_NONE)
            ctx.set_depth_test(False)
            ctx.set_depth_write(False)
            ctx.set_color_mask(True, True, True, True)
            ctx.set_blend(True)
            ctx.set_blend_func(Tgfx2BlendFactor.One,
                               Tgfx2BlendFactor.OneMinusSrcAlpha)
            ctx.bind_shader(self._composite_vs, self._composite_fs)

            self._draw_texture_quad(
                temp,
                layer.opacity,
                source_premultiplied=True,
            )
            self._release_temp_tex(temp)
        else:
            if has_children:
                for child in reversed(layer.children):
                    self._render_layer_tree(child, target_tex)
            if own_visible and layer.opacity > 0:
                self._draw_layer_quad(layer, layer.opacity)

    def _draw_layer_quad(self, layer: Layer, opacity: float):
        lid = id(layer)
        tex = self._layer_textures.get(lid)
        if tex is None:
            return
        x0, y0, x1, y1 = layer.bounds
        if x1 <= 0 or y1 <= 0 or x0 >= self._fbo_w or y0 >= self._fbo_h:
            return
        verts = _canvas_quad_verts(x0, y0, x1, y1, self._fbo_w, self._fbo_h)
        self._draw_texture_quad(tex, opacity, verts)

    def _draw_texture_quad(
            self,
            texture: Tgfx2TextureHandle,
            opacity: float,
            verts: np.ndarray | None = None,
            *,
            source_premultiplied: bool = False):
        ctx = self._ctx
        ctx.bind_texture_by_name("u_texture", texture)
        data = struct.pack(
            _COMPOSITE_PARAMS_FMT,
            float(opacity),
            1.0 if source_premultiplied else 0.0,
        )
        ctx.bind_uniform_by_name("u_composite", data)
        ctx.draw_immediate_triangles(self._quad_verts if verts is None else verts, 6)

    # ------------------------------------------------------------------
    # Temp texture pool
    # ------------------------------------------------------------------

    def _acquire_temp_tex(self) -> Tgfx2TextureHandle:
        if self._temp_texs_in_use < len(self._temp_texs):
            tex = self._temp_texs[self._temp_texs_in_use]
        else:
            tex = self._graphics.create_color_attachment(
                self._fbo_w, self._fbo_h, Tgfx2PixelFormat.RGBA16F)
            self._temp_texs.append(tex)
        self._temp_texs_in_use += 1
        return tex

    def _release_temp_tex(self, tex: Tgfx2TextureHandle):
        self._temp_texs_in_use = max(0, self._temp_texs_in_use - 1)
