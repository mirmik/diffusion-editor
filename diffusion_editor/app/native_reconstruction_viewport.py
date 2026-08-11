"""Native Termin viewport for generated GLB meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image
from tcbase import Action, MouseButton
from tcbase._geom_native import LinearColor
from termin.geombase import OrbitCamera, Vec3
from termin.gui_native import Size, TcDocument
from tgfx import (
    CULL_NONE,
    PIXEL_D32F,
    PIXEL_RGBA8,
    PointCloud,
    PointCloudDrawParams,
    PointCloudRenderer,
    PointCloudShape,
    PointCloudStyle,
    Tgfx2Context,
    Tgfx2ShaderStage,
    TextureEncoding,
    draw_tc_mesh,
)
from tgfx._tgfx_native import (
    ShaderArtifactPolicy,
    ShaderLanguage,
    TcShader,
    tc_shader_ensure_tgfx2,
)


_VERTEX_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_MVP pc.u_mvp
#else
uniform mat4 u_mvp;
#define U_MVP u_mvp
#endif
layout(location=0) in vec3 a_position;
layout(location=0) out vec3 v_position;
void main() {
    v_position = a_position;
    gl_Position = U_MVP * vec4(a_position, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_COLOR pc.u_color
#define U_LIGHT_DIRECTION pc.u_light_direction.xyz
#define U_DISPLAY_MODE pc.u_light_direction.w
#else
uniform vec4 u_color;
uniform vec4 u_light_direction;
#define U_COLOR u_color
#define U_LIGHT_DIRECTION u_light_direction.xyz
#define U_DISPLAY_MODE u_light_direction.w
#endif
layout(location=0) in vec3 v_position;
layout(location=0) out vec4 frag_color;
void main() {
    vec3 normal = normalize(cross(dFdy(v_position), dFdx(v_position)));
    int display_mode = int(U_DISPLAY_MODE + 0.5);
    if (display_mode == 2) {
        frag_color = U_COLOR;
        return;
    }
    if (display_mode == 3) {
        frag_color = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 light_direction = normalize(U_LIGHT_DIRECTION);
    float diffuse = max(dot(normal, light_direction), 0.0);
    vec3 color = U_COLOR.rgb * (0.28 + 0.72 * diffuse);
    frag_color = vec4(color, 1.0);
}
"""

_SMOOTH_VERTEX_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_MVP pc.u_mvp
#else
uniform mat4 u_mvp;
#define U_MVP u_mvp
#endif
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=0) out vec3 v_normal;
void main() {
    v_normal = a_normal;
    gl_Position = U_MVP * vec4(a_position, 1.0);
}
"""

_SMOOTH_FRAGMENT_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_COLOR pc.u_color
#define U_LIGHT_DIRECTION pc.u_light_direction.xyz
#define U_DISPLAY_MODE pc.u_light_direction.w
#else
uniform vec4 u_color;
uniform vec4 u_light_direction;
#define U_COLOR u_color
#define U_LIGHT_DIRECTION u_light_direction.xyz
#define U_DISPLAY_MODE u_light_direction.w
#endif
layout(location=0) in vec3 v_normal;
layout(location=0) out vec4 frag_color;
void main() {
    vec3 normal = normalize(v_normal);
    normal = gl_FrontFacing ? normal : -normal;
    int display_mode = int(U_DISPLAY_MODE + 0.5);
    if (display_mode == 2) {
        frag_color = U_COLOR;
        return;
    }
    if (display_mode == 3) {
        frag_color = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 light_direction = normalize(U_LIGHT_DIRECTION);
    float diffuse = max(dot(normal, light_direction), 0.0);
    vec3 color = U_COLOR.rgb * (0.28 + 0.72 * diffuse);
    frag_color = vec4(color, 1.0);
}
"""

_TEXTURED_VERTEX_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_MVP pc.u_mvp
#else
uniform mat4 u_mvp;
#define U_MVP u_mvp
#endif
layout(location=0) in vec3 a_position;
layout(location=2) in vec2 a_uv;
layout(location=0) out vec3 v_position;
layout(location=1) out vec2 v_uv;
void main() {
    v_position = a_position;
    v_uv = a_uv;
    gl_Position = U_MVP * vec4(a_position, 1.0);
}
"""

_TEXTURED_FRAGMENT_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_COLOR pc.u_color
#define U_LIGHT_DIRECTION pc.u_light_direction.xyz
#define U_DISPLAY_MODE pc.u_light_direction.w
#else
uniform vec4 u_color;
uniform vec4 u_light_direction;
#define U_COLOR u_color
#define U_LIGHT_DIRECTION u_light_direction.xyz
#define U_DISPLAY_MODE u_light_direction.w
#endif
layout(binding=0) uniform sampler2D u_base_color_texture;
layout(location=0) in vec3 v_position;
layout(location=1) in vec2 v_uv;
layout(location=0) out vec4 frag_color;
void main() {
    vec3 normal = normalize(cross(dFdy(v_position), dFdx(v_position)));
    vec4 base_color = texture(u_base_color_texture, v_uv) * U_COLOR;
    int display_mode = int(U_DISPLAY_MODE + 0.5);
    if (display_mode == 2) {
        frag_color = base_color;
        return;
    }
    if (display_mode == 3) {
        frag_color = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 light_direction = normalize(U_LIGHT_DIRECTION);
    float diffuse = max(dot(normal, light_direction), 0.0);
    frag_color = vec4(base_color.rgb * (0.28 + 0.72 * diffuse), base_color.a);
}
"""

_SMOOTH_TEXTURED_VERTEX_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_MVP pc.u_mvp
#else
uniform mat4 u_mvp;
#define U_MVP u_mvp
#endif
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_uv;
layout(location=0) out vec3 v_normal;
layout(location=1) out vec2 v_uv;
void main() {
    v_normal = a_normal;
    v_uv = a_uv;
    gl_Position = U_MVP * vec4(a_position, 1.0);
}
"""

_SMOOTH_TEXTURED_FRAGMENT_SHADER = """#version 450 core
#ifdef VULKAN
layout(push_constant) uniform PCBlock {
    mat4 u_mvp;
    vec4 u_color;
    vec4 u_light_direction;
} pc;
#define U_COLOR pc.u_color
#define U_LIGHT_DIRECTION pc.u_light_direction.xyz
#define U_DISPLAY_MODE pc.u_light_direction.w
#else
uniform vec4 u_color;
uniform vec4 u_light_direction;
#define U_COLOR u_color
#define U_LIGHT_DIRECTION u_light_direction.xyz
#define U_DISPLAY_MODE u_light_direction.w
#endif
layout(binding=0) uniform sampler2D u_base_color_texture;
layout(location=0) in vec3 v_normal;
layout(location=1) in vec2 v_uv;
layout(location=0) out vec4 frag_color;
void main() {
    vec3 normal = normalize(v_normal);
    normal = gl_FrontFacing ? normal : -normal;
    vec4 base_color = texture(u_base_color_texture, v_uv) * U_COLOR;
    int display_mode = int(U_DISPLAY_MODE + 0.5);
    if (display_mode == 2) {
        frag_color = base_color;
        return;
    }
    if (display_mode == 3) {
        frag_color = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 light_direction = normalize(U_LIGHT_DIRECTION);
    float diffuse = max(dot(normal, light_direction), 0.0);
    frag_color = vec4(base_color.rgb * (0.28 + 0.72 * diffuse), base_color.a);
}
"""


RECONSTRUCTION_SHADING_MODES = (
    "flat", "smooth", "unlit", "normals", "wireframe"
)
RECONSTRUCTION_SHADING_LABELS = {
    "flat": "Flat",
    "smooth": "Smooth",
    "unlit": "Unlit",
    "normals": "Normals",
    "wireframe": "Wireframe",
}
_SHADING_MODE_IDS = {
    "flat": 0.0,
    "smooth": 1.0,
    "unlit": 2.0,
    "normals": 3.0,
    "wireframe": 4.0,
}


@dataclass
class _MeshRenderItem:
    mesh: object
    texture: object | None = None
    color: tuple[float, float, float, float] = (0.34, 0.72, 0.95, 1.0)
    has_normals: bool = False
    wire_mesh: object | None = None


def _wireframe_indices(triangles: np.ndarray) -> np.ndarray:
    """Expand triangle indices into three independent line segments each."""
    faces = np.asarray(triangles, dtype=np.uint32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must have shape (count, 3)")
    if not len(faces):
        return np.empty(0, dtype=np.uint32)
    result = np.empty((len(faces), 6), dtype=np.uint32)
    result[:, 0:2] = faces[:, (0, 1)]
    result[:, 2:4] = faces[:, (1, 2)]
    result[:, 4:6] = faces[:, (2, 0)]
    return np.ascontiguousarray(result.reshape(-1))


def _build_wireframe_mesh(source_mesh):
    from tmesh import TcAttribType, TcDrawMode, TcMesh, TcVertexLayout

    positions = np.ascontiguousarray(source_mesh.vertices, dtype=np.float32)
    triangles = source_mesh.triangles
    if triangles is None or not len(triangles):
        return None
    indices = _wireframe_indices(triangles)
    layout = TcVertexLayout()
    if not layout.add("position", 3, TcAttribType.FLOAT32, 0):
        raise RuntimeError("failed to create reconstruction wireframe layout")
    return TcMesh.from_interleaved(
        positions,
        len(positions),
        indices,
        layout,
        name=f"{source_mesh.name} Wireframe",
        uuid=f"{source_mesh.uuid}-wireframe",
        draw_mode=TcDrawMode.LINES,
    )


def _decode_texture(payload: bytes) -> tuple[int, int, np.ndarray]:
    with Image.open(BytesIO(payload)) as image:
        # nanobind's writable uint8 ndarray boundary rejects Pillow's
        # read-only array view, even when it is otherwise C-contiguous.
        rgba = np.array(image.convert("RGBA"), dtype=np.uint8, order="C", copy=True)
    height, width = rgba.shape[:2]
    return width, height, rgba.reshape(-1).copy()


def _should_fit_camera(had_model: bool, requested: bool | None) -> bool:
    """Fit a first model by default, while preserving an existing comparison view."""
    return not had_model if requested is None else bool(requested)


class _OrbitCamera:
    FRONT_AZIMUTH = math.pi
    FRONT_ELEVATION = math.radians(12.0)

    def __init__(self) -> None:
        self._camera = OrbitCamera()
        self._camera.target = Vec3(0.0, 0.0, 0.0)
        self._camera.distance = 3.0
        self._reset_orientation()
        self._camera.fov_y = 0.78
        self._camera.near = 0.01
        self._camera.far = 100.0

    def orbit(self, dx: float, dy: float) -> None:
        self._camera.orbit(-float(dx) * 0.01, float(dy) * 0.01)

    def pan(self, dx: float, dy: float) -> None:
        self._camera.pan(float(dx), float(dy))

    def zoom(self, delta: float) -> None:
        self._camera.zoom(max(0.15, 1.0 - float(delta) * 0.1))

    def fit(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        center = (minimum + maximum) * 0.5
        radius = max(float(np.linalg.norm(maximum - minimum)) * 0.65, 0.25)
        self._camera.target = Vec3(*map(float, center))
        self._camera.distance = radius * 2.7
        self._camera.near = max(radius * 0.001, 0.001)
        self._camera.far = max(radius * 20.0, 10.0)
        self._reset_orientation()

    def direction_from_target(self) -> np.ndarray:
        eye = np.asarray(tuple(self._camera.eye), dtype=np.float32)
        target = np.asarray(tuple(self._camera.target), dtype=np.float32)
        direction = eye - target
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        return direction / length

    def mvp(self, width: int, height: int) -> np.ndarray:
        aspect = max(float(width) / max(float(height), 1.0), 0.001)
        flat = np.asarray(self._camera.mvp(aspect), dtype=np.float32)
        return flat.reshape((4, 4), order="F")

    def _reset_orientation(self) -> None:
        # Verified against generated Pixal3D subjects: after the Z-up export
        # conversion their visible front is viewed from +Y.
        self._camera.azimuth = self.FRONT_AZIMUTH
        self._camera.elevation = self.FRONT_ELEVATION


def _draw_constants(
    mvp: np.ndarray,
    color,
    light_direction,
    display_mode: float = 0.0,
) -> np.ndarray:
    gpu_mvp = np.ascontiguousarray(mvp.T, dtype=np.float32)
    color_array = np.asarray(color, dtype=np.float32)
    light_array = np.asarray(
        (*light_direction, float(display_mode)), dtype=np.float32
    )
    constants = np.concatenate(
        (gpu_mvp.reshape(-1), color_array, light_array)
    )
    return np.ascontiguousarray(constants.view(np.uint8), dtype=np.uint8)


class _ViewportSurface:
    def __init__(self, camera: _OrbitCamera, on_changed: Callable[[], None]) -> None:
        self._camera = camera
        self._on_changed = on_changed
        self._valid = True
        self._width = 1
        self._height = 1
        self._texture = None
        self._pointer = (0.0, 0.0)
        self._drag_mode = ""

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    def publish_texture(self, texture) -> None:
        self._texture = texture

    def close(self) -> None:
        self._valid = False
        self._texture = None

    def is_valid(self) -> bool:
        return self._valid

    def get_tgfx_color_tex_id(self) -> int:
        return 0 if self._texture is None else int(self._texture.id)

    def framebuffer_size(self) -> tuple[int, int]:
        return self.size

    def resize(self, width: int, height: int) -> bool:
        next_size = (max(int(width), 1), max(int(height), 1))
        if not self._valid:
            return False
        if next_size != self.size:
            self._width, self._height = next_size
            self._texture = None
            self._on_changed()
        return True

    def dispatch_pointer_move(self, x: float, y: float) -> bool:
        if not self._valid:
            return False
        previous = self._pointer
        self._pointer = (float(x), float(y))
        if not self._drag_mode:
            return False
        dx = self._pointer[0] - previous[0]
        dy = self._pointer[1] - previous[1]
        if self._drag_mode == "orbit":
            self._camera.orbit(dx, dy)
        else:
            self._camera.pan(-dx, dy)
        self._on_changed()
        return True

    def dispatch_pointer_button(
        self,
        x: float,
        y: float,
        button: int,
        action: int,
        modifiers: int,
        click_count: int,
    ) -> bool:
        del modifiers, click_count
        if not self._valid:
            return False
        self._pointer = (float(x), float(y))
        if int(action) == int(Action.RELEASE):
            self._drag_mode = ""
            return True
        if int(action) != int(Action.PRESS):
            return False
        if int(button) == int(MouseButton.LEFT):
            self._drag_mode = "orbit"
        elif int(button) in {int(MouseButton.MIDDLE), int(MouseButton.RIGHT)}:
            self._drag_mode = "pan"
        else:
            return False
        return True

    def dispatch_wheel(
        self,
        x: float,
        y: float,
        wheel_x: float,
        wheel_y: float,
        modifiers: int,
    ) -> bool:
        del x, y, wheel_x, modifiers
        if not self._valid:
            return False
        self._camera.zoom(float(wheel_y))
        self._on_changed()
        return True

    def dispatch_key(self, key: int, scancode: int, action: int, modifiers: int) -> bool:
        del key, scancode, action, modifiers
        return False

    def dispatch_text(self, codepoint: int) -> bool:
        del codepoint
        return False


class NativeReconstructionViewport:
    """Displays a generated mesh or point cloud with orbit, pan and zoom."""

    def __init__(
        self,
        document: TcDocument,
        *,
        graphics_owner,
        request_repaint: Callable[[], None],
    ) -> None:
        self._graphics = Tgfx2Context.from_runtime(graphics_owner)
        self._request_repaint = request_repaint
        self._camera = _OrbitCamera()
        self._light_direction = self._camera.direction_from_target()
        self._shading_mode = "flat"
        self._dirty = True
        self._closed = False
        self._mesh_items: list[_MeshRenderItem] = []
        self._point_cloud = PointCloud()
        self._point_cloud_renderer = PointCloudRenderer()
        self._point_cloud_style = PointCloudStyle()
        self._point_cloud_style.size_px = 5.0
        self._point_cloud_style.shape = PointCloudShape.Circle
        self._point_cloud_draw_params = PointCloudDrawParams()
        self._glb_document = None
        self._color_texture = None
        self._depth_texture = None
        self._texture_size = (0, 0)
        self._vertex_shader = self._graphics.device.create_shader(
            Tgfx2ShaderStage.Vertex, _VERTEX_SHADER
        )
        self._fragment_shader = self._graphics.device.create_shader(
            Tgfx2ShaderStage.Fragment, _FRAGMENT_SHADER
        )
        self._smooth_vertex_shader = self._graphics.device.create_shader(
            Tgfx2ShaderStage.Vertex, _SMOOTH_VERTEX_SHADER
        )
        self._smooth_fragment_shader = self._graphics.device.create_shader(
            Tgfx2ShaderStage.Fragment, _SMOOTH_FRAGMENT_SHADER
        )
        self._textured_shader = TcShader.get_or_create(
            "diffusion-editor-reconstruction-textured"
        )
        self._textured_shader.set_language(ShaderLanguage.GLSL)
        self._textured_shader.set_artifact_policy(ShaderArtifactPolicy.REQUIRED)
        self._textured_shader.set_sources_with_entries(
            _TEXTURED_VERTEX_SHADER,
            _TEXTURED_FRAGMENT_SHADER,
            "",
            "diffusion-editor-reconstruction-textured",
            "diffusion_editor/app/reconstruction_textured.glsl",
            "main",
            "main",
        )
        textured_pair = tc_shader_ensure_tgfx2(
            self._graphics.context, self._textured_shader
        )
        self._textured_vertex_shader = textured_pair.vs
        self._textured_fragment_shader = textured_pair.fs
        if not self._textured_vertex_shader or not self._textured_fragment_shader:
            raise RuntimeError("Reconstruction texture shader compile failed")
        self._smooth_textured_shader = TcShader.get_or_create(
            "diffusion-editor-reconstruction-smooth-textured"
        )
        self._smooth_textured_shader.set_language(ShaderLanguage.GLSL)
        self._smooth_textured_shader.set_artifact_policy(
            ShaderArtifactPolicy.REQUIRED
        )
        self._smooth_textured_shader.set_sources_with_entries(
            _SMOOTH_TEXTURED_VERTEX_SHADER,
            _SMOOTH_TEXTURED_FRAGMENT_SHADER,
            "",
            "diffusion-editor-reconstruction-smooth-textured",
            "diffusion_editor/app/reconstruction_smooth_textured.glsl",
            "main",
            "main",
        )
        smooth_textured_pair = tc_shader_ensure_tgfx2(
            self._graphics.context, self._smooth_textured_shader
        )
        self._smooth_textured_vertex_shader = smooth_textured_pair.vs
        self._smooth_textured_fragment_shader = smooth_textured_pair.fs
        if (
            not self._smooth_textured_vertex_shader
            or not self._smooth_textured_fragment_shader
        ):
            raise RuntimeError(
                "Reconstruction smooth texture shader compile failed"
            )
        self.surface = _ViewportSurface(self._camera, self.invalidate)
        self.viewport = document.create_viewport3d()
        self.viewport.widget.stable_id = "diffusion-editor.reconstruction.viewport"
        self.viewport.widget.preferred_size = Size(480.0, 600.0)
        self.viewport.set_surface_host(self.surface)
        self.widget = self.viewport.widget

    @property
    def mesh_count(self) -> int:
        return len(self._mesh_items)

    @property
    def point_count(self) -> int:
        return int(self._point_cloud.point_count)

    @property
    def light_direction(self) -> tuple[float, float, float]:
        return tuple(map(float, self._light_direction))

    @property
    def shading_mode(self) -> str:
        return self._shading_mode

    def set_shading_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in RECONSTRUCTION_SHADING_MODES:
            raise ValueError(f"unsupported reconstruction shading mode: {mode}")
        if normalized == self._shading_mode:
            return
        self._shading_mode = normalized
        self.invalidate()

    def light_from_camera(self) -> None:
        self._require_open()
        self._light_direction = self._camera.direction_from_target()
        self.invalidate()

    def clear_model(self) -> None:
        self._require_open()
        self._destroy_model_textures()
        self._mesh_items.clear()
        self._point_cloud.release(self._graphics.context)
        self._glb_document = None
        self.invalidate()

    def load_glb(
            self, path: str, *, fit_camera: bool | None = None
    ) -> tuple[int, int, int]:
        self._require_open()
        from ..native_glb import NativeGLBDocument

        had_model = bool(self._mesh_items) or not self._point_cloud.empty
        source = NativeGLBDocument(path)
        mesh_items = []
        vertices = 0
        indices = 0
        bounds_min = None
        bounds_max = None
        for index, info in enumerate(source.meshes):
            if info.skinned:
                raise RuntimeError("The first reconstruction viewport supports static GLB meshes only")
            mesh = source.build_mesh(
                index,
                f"diffusion-editor-reconstruction-mesh-{index}",
                name=info.name or f"Reconstruction mesh {index}",
                convert_to_z_up=True,
            )
            texture = None
            color = (0.34, 0.72, 0.95, 1.0)
            base_color = source.base_color_texture(index)
            if base_color is not None:
                width, height, pixels = _decode_texture(base_color.payload)
                texture = self._graphics.create_texture_rgba8(
                    width, height, pixels, TextureEncoding.SRGB
                )
                color = base_color.factor
            mesh_items.append(_MeshRenderItem(
                mesh,
                texture,
                color,
                has_normals=mesh.vertex_normals is not None,
            ))
            vertices += int(mesh.vertex_count)
            indices += int(mesh.index_count)
            payload = np.asarray(mesh.mesh.get_vertices_buffer(), dtype=np.float32)
            stride = int(mesh.stride) // 4
            if stride >= 3 and payload.size >= stride:
                positions = payload.reshape(-1, stride)[:, :3]
                local_min = positions.min(axis=0)
                local_max = positions.max(axis=0)
                bounds_min = local_min if bounds_min is None else np.minimum(bounds_min, local_min)
                bounds_max = local_max if bounds_max is None else np.maximum(bounds_max, local_max)
        if not mesh_items:
            raise RuntimeError("Generated GLB contains no meshes")
        self._destroy_model_textures()
        self._point_cloud.release(self._graphics.context)
        self._glb_document = source
        self._mesh_items = mesh_items
        if (
            bounds_min is not None
            and bounds_max is not None
            and _should_fit_camera(had_model, fit_camera)
        ):
            self._camera.fit(bounds_min, bounds_max)
            self._light_direction = self._camera.direction_from_target()
        self.invalidate()
        return vertices, indices // 3, len(mesh_items)

    def load_point_cloud(
            self, path: str, *, fit_camera: bool | None = None
    ) -> int:
        self._require_open()
        from ..native_ply import load_ply_points

        had_model = bool(self._mesh_items) or not self._point_cloud.empty
        data = load_ply_points(path)
        if not self._point_cloud.upload_srgb(
            self._graphics.context, data.positions, data.colors
        ):
            raise RuntimeError("Termin failed to upload the reconstruction point cloud")
        self._destroy_model_textures()
        self._mesh_items.clear()
        self._glb_document = None
        if _should_fit_camera(had_model, fit_camera):
            self._camera.fit(data.positions.min(axis=0), data.positions.max(axis=0))
            self._light_direction = self._camera.direction_from_target()
        self.invalidate()
        return len(data.positions)

    def invalidate(self) -> None:
        if self._closed:
            return
        self._dirty = True
        self._request_repaint()

    def render_if_dirty(self) -> bool:
        if self._closed or not self._dirty:
            return False
        if self._shading_mode == "wireframe":
            for item in self._mesh_items:
                if item.wire_mesh is None:
                    item.wire_mesh = _build_wireframe_mesh(item.mesh)
        width, height = self.surface.size
        self._ensure_textures(width, height)
        ctx = self._graphics.context
        opened_frame = not ctx.in_frame
        if opened_frame:
            ctx.begin_frame()
        ctx.begin_pass(
            self._color_texture,
            self._depth_texture,
            clear_linear_color=LinearColor(0.055, 0.06, 0.075, 1.0),
            clear_depth=1.0,
            clear_depth_enabled=True,
        )
        ctx.set_viewport(0, 0, width, height)
        ctx.set_depth_test(True)
        ctx.set_depth_write(True)
        ctx.set_blend(False)
        ctx.set_cull(CULL_NONE)
        mvp = self._camera.mvp(width, height)
        display_mode = _SHADING_MODE_IDS[self._shading_mode]
        for item in self._mesh_items:
            if self._shading_mode == "wireframe":
                if item.wire_mesh is None:
                    continue
                ctx.bind_shader(self._vertex_shader, self._fragment_shader)
                self._set_draw_state(
                    mvp,
                    (0.72, 0.80, 0.95, 1.0),
                    self._light_direction,
                    _SHADING_MODE_IDS["unlit"],
                )
                draw_tc_mesh(ctx, item.wire_mesh)
                continue
            use_smooth_normals = (
                self._shading_mode in {"smooth", "normals"}
                and item.has_normals
            )
            if item.texture is None and use_smooth_normals:
                ctx.bind_shader(
                    self._smooth_vertex_shader, self._smooth_fragment_shader
                )
            elif item.texture is None:
                ctx.bind_shader(self._vertex_shader, self._fragment_shader)
            elif use_smooth_normals:
                ctx.bind_shader(
                    self._smooth_textured_vertex_shader,
                    self._smooth_textured_fragment_shader,
                )
                ctx.use_shader_resource_layout(self._smooth_textured_shader)
                ctx.bind_texture_by_name("u_base_color_texture", item.texture)
            else:
                ctx.bind_shader(
                    self._textured_vertex_shader, self._textured_fragment_shader
                )
                ctx.use_shader_resource_layout(self._textured_shader)
                ctx.bind_texture_by_name("u_base_color_texture", item.texture)
            self._set_draw_state(
                mvp,
                item.color,
                self._light_direction,
                display_mode,
            )
            draw_tc_mesh(ctx, item.mesh)
        if not self._point_cloud.empty:
            self._point_cloud_draw_params.view_projection = tuple(
                np.ascontiguousarray(mvp.T, dtype=np.float32).reshape(-1)
            )
            self._point_cloud_renderer.draw(
                ctx,
                self._point_cloud,
                self._point_cloud_style,
                self._point_cloud_draw_params,
            )
        ctx.end_pass()
        if opened_frame:
            ctx.end_frame()
        self.surface.publish_texture(self._color_texture)
        self._dirty = False
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.viewport.detach_surface()
        self.surface.close()
        if self._color_texture is not None:
            self._graphics.destroy_texture(self._color_texture)
        if self._depth_texture is not None:
            self._graphics.destroy_texture(self._depth_texture)
        self._graphics.device.destroy_shader(self._vertex_shader)
        self._graphics.device.destroy_shader(self._fragment_shader)
        self._graphics.device.destroy_shader(self._smooth_vertex_shader)
        self._graphics.device.destroy_shader(self._smooth_fragment_shader)
        self._point_cloud.release(self._graphics.context)
        self._point_cloud_renderer.release(self._graphics.context)
        self._color_texture = None
        self._depth_texture = None
        self._destroy_model_textures()
        self._mesh_items.clear()
        self._glb_document = None

    def _destroy_model_textures(self) -> None:
        for item in self._mesh_items:
            if item.texture is not None:
                self._graphics.destroy_texture(item.texture)

    def _ensure_textures(self, width: int, height: int) -> None:
        size = (max(int(width), 1), max(int(height), 1))
        if size == self._texture_size:
            return
        if self._color_texture is not None:
            self._graphics.destroy_texture(self._color_texture)
        if self._depth_texture is not None:
            self._graphics.destroy_texture(self._depth_texture)
        self._color_texture = self._graphics.create_color_attachment(
            *size, PIXEL_RGBA8
        )
        self._depth_texture = self._graphics.create_depth_attachment(
            *size, PIXEL_D32F
        )
        self._texture_size = size

    def _set_draw_state(
        self,
        mvp: np.ndarray,
        color,
        light_direction,
        display_mode: float,
    ) -> None:
        self._graphics.context.set_push_constants(
            _draw_constants(mvp, color, light_direction, display_mode)
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reconstruction viewport is closed")
