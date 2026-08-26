"""Native Termin viewport for generated GLB meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from itertools import count
import math

import numpy as np
from PIL import Image
from tcbase import Action, MouseButton
from tcbase._geom_native import LinearColor
from termin.geombase import OrbitCamera, Rect2, Vec2, Vec3
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

from ..generation.local_detail_geometry import refine_placement_matrix
from ..generation.types import ReconstructionRefinePlacement


_VIEWPORT_RESOURCE_IDS = count()


def _allocate_resource_namespace(label: str) -> str:
    return f"{str(label)}-{next(_VIEWPORT_RESOURCE_IDS)}"


def _mesh_resource_id(namespace: str, mesh_index: int) -> str:
    return f"diffusion-editor-reconstruction-{namespace}-mesh-{mesh_index}"


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
    float diffuse = abs(dot(normal, light_direction));
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
    float diffuse = abs(dot(normal, light_direction));
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
POINT_CLOUD_COLOR_MODES = ("image", "confidence")
POINT_CLOUD_COLOR_LABELS = {
    "image": "Image",
    "confidence": "Confidence",
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
    smooth_mesh: object | None = None
    wire_mesh: object | None = None
    editable: bool = True
    positions: np.ndarray | None = None
    triangles: np.ndarray | None = None


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


def _nearest_ray_triangle_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    positions: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray | None:
    """Return the nearest forward Moller-Trumbore hit, if any."""
    vertices = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if not len(vertices) or not len(faces):
        return None
    ray_origin = np.asarray(origin, dtype=np.float64)
    ray_direction = np.asarray(direction, dtype=np.float64)
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    cross = np.cross(np.broadcast_to(ray_direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross)
    usable = np.abs(determinant) > 1e-10
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    offset = ray_origin - v0
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    cross_offset = np.cross(offset, edge1)
    v = np.einsum("j,ij->i", ray_direction, cross_offset) * inverse
    distance = np.einsum("ij,ij->i", edge2, cross_offset) * inverse
    hits = usable & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance > 1e-8)
    if not np.any(hits):
        return None
    nearest = float(np.min(distance[hits]))
    return np.asarray(ray_origin + ray_direction * nearest, dtype=np.float32)


def _build_unit_cube_meshes(namespace: str):
    from tmesh import TcAttribType, TcDrawMode, TcMesh, TcVertexLayout

    positions = np.ascontiguousarray([
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ], dtype=np.float32)
    triangles = np.ascontiguousarray([
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    ], dtype=np.uint32)
    lines = np.ascontiguousarray([
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
    ], dtype=np.uint32)
    layout = TcVertexLayout()
    if not layout.add("position", 3, TcAttribType.FLOAT32, 0):
        raise RuntimeError("failed to create refine cube layout")
    fill = TcMesh.from_interleaved(
        positions, len(positions), triangles, layout,
        name="Refine cube fill", uuid=f"{namespace}-refine-cube-fill",
    )
    wire = TcMesh.from_interleaved(
        positions, len(positions), lines, layout,
        name="Refine cube wire", uuid=f"{namespace}-refine-cube-wire",
        draw_mode=TcDrawMode.LINES,
    )
    return fill, wire


def _build_unit_arrow_mesh(namespace: str):
    from tmesh import TcAttribType, TcMesh, TcVertexLayout

    positions = np.ascontiguousarray([
        (0.00, -0.018, -0.018), (0.00, 0.018, -0.018),
        (0.00, 0.018, 0.018), (0.00, -0.018, 0.018),
        (0.76, -0.018, -0.018), (0.76, 0.018, -0.018),
        (0.76, 0.018, 0.018), (0.76, -0.018, 0.018),
        (0.74, -0.065, -0.065), (0.74, 0.065, -0.065),
        (0.74, 0.065, 0.065), (0.74, -0.065, 0.065),
        (1.00, 0.00, 0.00),
    ], dtype=np.float32)
    triangles = np.ascontiguousarray([
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
        8, 9, 12, 9, 10, 12, 10, 11, 12, 11, 8, 12,
        8, 11, 10, 8, 10, 9,
    ], dtype=np.uint32)
    layout = TcVertexLayout()
    if not layout.add("position", 3, TcAttribType.FLOAT32, 0):
        raise RuntimeError("failed to create refine gizmo arrow layout")
    return TcMesh.from_interleaved(
        positions,
        len(positions),
        triangles,
        layout,
        name="Refine gizmo arrow",
        uuid=f"{namespace}-refine-gizmo-arrow",
    )


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    segment = np.asarray(end, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(np.asarray(point) - start)), 0.0
    parameter = float(np.dot(np.asarray(point) - start, segment) / length_squared)
    parameter = min(max(parameter, 0.0), 1.0)
    closest = np.asarray(start) + segment * parameter
    return float(np.linalg.norm(np.asarray(point) - closest)), parameter


def _ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    denominator = float(np.dot(direction, plane_normal))
    if abs(denominator) <= 1e-8:
        return None
    distance = float(np.dot(plane_point - origin, plane_normal) / denominator)
    if not math.isfinite(distance):
        return None
    return np.asarray(origin + direction * distance, dtype=np.float64)


def _oriented_transform(
    origin: np.ndarray, direction: np.ndarray, length: float
) -> np.ndarray:
    axis = np.asarray(direction, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    helper = (
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(axis[2])) < 0.9
        else np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    )
    local_y = np.cross(helper, axis)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(axis, local_y)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 0] = axis * float(length)
    transform[:3, 1] = local_y * float(length)
    transform[:3, 2] = local_z * float(length)
    transform[:3, 3] = np.asarray(origin, dtype=np.float32)
    return transform


_REFINE_GIZMO_AXES = {
    # Labels and colors follow the persistent glTF Y-up coordinates. Termin's
    # preview converts them to Z-up: (x, y, z) -> (x, -z, y).
    "x": (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        (0.95, 0.16, 0.12, 1.0),
    ),
    "y": (
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        (0.24, 0.88, 0.28, 1.0),
    ),
    "z": (
        np.asarray((0.0, -1.0, 0.0), dtype=np.float64),
        (0.18, 0.42, 1.0, 1.0),
    ),
}
_REFINE_SIZE_DIRECTION = np.asarray(
    (1.0, -1.0, 1.0), dtype=np.float64
) / math.sqrt(3.0)


@dataclass
class _RefineGizmoDrag:
    kind: str
    original_center: np.ndarray
    original_side: float
    constraint_direction: np.ndarray
    plane_normal: np.ndarray
    start_parameter: float


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


def _compute_vertex_normals(
    positions: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Return area-weighted smooth normals for an indexed triangle mesh."""
    vertices = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.uint32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    if not len(vertices) or not len(faces):
        return normals
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-12
    normals[usable] /= lengths[usable, None]
    normals[~usable] = (0.0, 0.0, 1.0)
    return np.ascontiguousarray(normals, dtype=np.float32)


def _build_smooth_mesh(source_mesh):
    """Add generated normals to a position-only mesh for legacy GLBs."""
    from tmesh import TcAttribType, TcMesh, TcVertexLayout

    positions = np.ascontiguousarray(source_mesh.vertices, dtype=np.float32)
    triangles = source_mesh.triangles
    if triangles is None or not len(triangles):
        return None
    faces = np.ascontiguousarray(triangles, dtype=np.uint32)
    normals = _compute_vertex_normals(positions, faces)
    vertices = np.ascontiguousarray(
        np.concatenate((positions, normals), axis=1), dtype=np.float32
    )
    layout = TcVertexLayout()
    if not layout.add("position", 3, TcAttribType.FLOAT32, 0):
        raise RuntimeError("failed to create smooth reconstruction position")
    if not layout.add("normal", 3, TcAttribType.FLOAT32, 12):
        raise RuntimeError("failed to create smooth reconstruction normal")
    return TcMesh.from_interleaved(
        vertices,
        len(vertices),
        np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32),
        layout,
        name=f"{source_mesh.name} Smooth",
        uuid=f"{source_mesh.uuid}-smooth",
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

    def __init__(self, front_azimuth: float = FRONT_AZIMUTH) -> None:
        self._camera = OrbitCamera()
        self._front_azimuth = float(front_azimuth)
        self._camera.target = Vec3(0.0, 0.0, 0.0)
        self._camera.distance = 3.0
        self._reset_orientation()
        self._camera.fov_y = 0.78
        self._camera.near = 0.01
        self._camera.far = 100.0

    def orbit(self, dx: float, dy: float) -> None:
        self._camera.orbit(-float(dx) * 0.01, float(dy) * 0.01)

    def begin_pan(
        self,
        position: tuple[float, float],
        width: int,
        height: int,
    ):
        return self._camera.begin_pan(
            Vec2(*map(float, position)),
            Rect2(0.0, 0.0, max(float(width), 1.0), max(float(height), 1.0)),
        )

    def pan(self, gesture, position: tuple[float, float]) -> bool:
        return bool(self._camera.pan(
            gesture,
            Vec2(*map(float, position)),
        ))

    def zoom(self, delta: float) -> None:
        self._camera.zoom(max(0.15, 1.0 - float(delta) * 0.1))

    def fit(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        center = (minimum + maximum) * 0.5
        radius = max(float(np.linalg.norm(maximum - minimum)) * 0.65, 0.25)
        self._camera.target = Vec3(*map(float, center))
        self._camera.distance = radius * 2.7
        # OrbitCamera recomputes its clipping planes from fitted_radius on
        # every zoom.  Keep it in sync with this cloud; otherwise the native
        # default radius can clip a large reconstruction after the first
        # wheel event.
        self._camera.fitted_radius = radius
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
        native_mvp = self._camera.mvp(aspect)
        # Current Termin returns a native Mat44.  Older SDKs exposed the
        # same column-major values as a flat tuple, so retain that fallback
        # while using the matrix's explicit storage adapter when available.
        values = (
            native_mvp.to_column_major()
            if hasattr(native_mvp, "to_column_major")
            else native_mvp
        )
        flat = np.asarray(values, dtype=np.float32)
        return flat.reshape((4, 4), order="F")

    def screen_ray(self, position: tuple[float, float], width: int, height: int):
        return self._camera.try_screen_ray(
            Vec2(*map(float, position)),
            Rect2(0.0, 0.0, max(float(width), 1.0), max(float(height), 1.0)),
        )

    def project_world_to_screen(
        self, point: np.ndarray, width: int, height: int
    ) -> np.ndarray | None:
        projected = self._camera.try_project_world_point(
            Vec3(*map(float, point)),
            Rect2(0.0, 0.0, max(float(width), 1.0), max(float(height), 1.0)),
        )
        if projected is None:
            return None
        return np.asarray(tuple(projected.screen), dtype=np.float64)

    def _reset_orientation(self) -> None:
        # Verified against generated Pixal3D subjects: after the Z-up export
        # conversion their visible front is viewed from +Y.
        self._camera.azimuth = self._front_azimuth
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
    def __init__(
        self,
        camera: _OrbitCamera,
        on_changed: Callable[[], None],
    ) -> None:
        self._camera = camera
        self._on_changed = on_changed
        self._key_handler: Callable[[int, int, int, int], bool] | None = None
        self._pointer_press_handler: Callable[[float, float, int], bool] | None = None
        self._pointer_move_handler: Callable[[float, float], bool] | None = None
        self._pointer_release_handler: Callable[[float, float, int], bool] | None = None
        self._valid = True
        self._width = 1
        self._height = 1
        self._texture = None
        self._pointer = (0.0, 0.0)
        self._drag_mode = ""
        self._pan_gesture = None

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    def publish_texture(self, texture) -> None:
        self._texture = texture

    def close(self) -> None:
        self._valid = False
        self._texture = None
        self._key_handler = None
        self._pointer_press_handler = None
        self._pointer_move_handler = None
        self._pointer_release_handler = None

    def set_key_handler(
        self,
        handler: Callable[[int, int, int, int], bool] | None,
    ) -> None:
        self._key_handler = handler

    def set_pointer_press_handler(
        self, handler: Callable[[float, float, int], bool] | None
    ) -> None:
        self._pointer_press_handler = handler

    def set_pointer_move_handler(
        self, handler: Callable[[float, float], bool] | None
    ) -> None:
        self._pointer_move_handler = handler

    def set_pointer_release_handler(
        self, handler: Callable[[float, float, int], bool] | None
    ) -> None:
        self._pointer_release_handler = handler

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
        if (
            self._pointer_move_handler is not None
            and self._pointer_move_handler(self._pointer[0], self._pointer[1])
        ):
            return True
        if not self._drag_mode:
            return False
        if self._drag_mode == "handled":
            return True
        dx = self._pointer[0] - previous[0]
        dy = self._pointer[1] - previous[1]
        if self._drag_mode == "orbit":
            self._camera.orbit(dx, dy)
        else:
            if self._pan_gesture is None:
                return False
            if not self._camera.pan(self._pan_gesture, self._pointer):
                return False
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
            if self._pointer_release_handler is not None:
                self._pointer_release_handler(
                    self._pointer[0], self._pointer[1], int(button)
                )
            self._drag_mode = ""
            self._pan_gesture = None
            return True
        if int(action) != int(Action.PRESS):
            return False
        if int(button) == int(MouseButton.LEFT):
            if (
                self._pointer_press_handler is not None
                and self._pointer_press_handler(
                    self._pointer[0], self._pointer[1], int(button)
                )
            ):
                self._drag_mode = "handled"
                return True
            self._drag_mode = "orbit"
        elif int(button) in {int(MouseButton.MIDDLE), int(MouseButton.RIGHT)}:
            self._pan_gesture = self._camera.begin_pan(
                self._pointer, self._width, self._height
            )
            if self._pan_gesture is None:
                return False
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
        if not self._valid or self._key_handler is None:
            return False
        return bool(self._key_handler(key, scancode, action, modifiers))

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
        resource_namespace: str = "viewport",
        front_azimuth: float = _OrbitCamera.FRONT_AZIMUTH,
        point_color_state_changed: (
            Callable[[bool, str, str], None] | None
        ) = None,
    ) -> None:
        self._resource_namespace = _allocate_resource_namespace(
            resource_namespace
        )
        self._graphics = Tgfx2Context.from_runtime(graphics_owner)
        self._request_repaint = request_repaint
        self._point_color_state_changed = point_color_state_changed
        self._camera = _OrbitCamera(front_azimuth)
        self._light_direction = self._camera.direction_from_target()
        self._shading_mode = "flat"
        self._dirty = True
        self._closed = False
        self._mesh_items: list[_MeshRenderItem] = []
        self._model_bounds_min: np.ndarray | None = None
        self._model_bounds_max: np.ndarray | None = None
        self._refine_cube_center: tuple[float, float, float] | None = None
        self._refine_cube_side = 0.0
        self._refine_cube_confirmed = False
        self._refine_cube_pick_handler: (
            Callable[[tuple[float, float, float], float], None] | None
        ) = None
        self._refine_cube_edit_handler: (
            Callable[[tuple[float, float, float], float], None] | None
        ) = None
        self._refine_cube_edit_enabled = True
        self._refine_gizmo_drag: _RefineGizmoDrag | None = None
        self._model_transform = np.eye(4, dtype=np.float32)
        self._refine_placement = ReconstructionRefinePlacement()
        self._refine_placement_pivot = (0.0, 0.0, 0.0)
        self._refine_placement_handler: (
            Callable[[ReconstructionRefinePlacement], None] | None
        ) = None
        self._point_cloud = PointCloud()
        self._point_cloud_positions: np.ndarray | None = None
        self._point_cloud_image_colors: np.ndarray | None = None
        self._point_cloud_confidence_colors: np.ndarray | None = None
        self._point_cloud_confidence_legend = ""
        self._point_cloud_color_mode = "image"
        self._point_cloud_renderer = PointCloudRenderer()
        self._point_cloud_style = PointCloudStyle()
        self._point_cloud_style.size_px = 5.0
        self._point_cloud_style.shape = PointCloudShape.Circle
        self._point_cloud_draw_params = PointCloudDrawParams()
        self._glb_documents = []
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
        self._refine_cube_fill_mesh, self._refine_cube_wire_mesh = (
            _build_unit_cube_meshes(self._resource_namespace)
        )
        self._refine_gizmo_arrow_mesh = _build_unit_arrow_mesh(
            self._resource_namespace
        )
        self.surface = _ViewportSurface(self._camera, self.invalidate)
        self.surface.set_pointer_press_handler(self._handle_pointer_press)
        self.surface.set_pointer_move_handler(self._handle_gizmo_pointer_move)
        self.surface.set_pointer_release_handler(
            self._handle_gizmo_pointer_release
        )
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

    @property
    def point_cloud_color_mode(self) -> str:
        return self._point_cloud_color_mode

    @property
    def point_cloud_has_confidence(self) -> bool:
        return self._point_cloud_confidence_colors is not None

    @property
    def point_cloud_size(self) -> float:
        return float(self._point_cloud_style.size_px)

    def set_point_cloud_size(self, size_px: float) -> None:
        self._require_open()
        value = float(size_px)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("point-cloud size must be finite and positive")
        if value == self._point_cloud_style.size_px:
            return
        self._point_cloud_style.size_px = value
        self.invalidate()

    def set_key_handler(
        self,
        handler: Callable[[int, int, int, int], bool] | None,
    ) -> None:
        """Handle keys routed directly to this viewport after it gains focus."""

        self._require_open()
        self.surface.set_key_handler(handler)

    def set_refine_cube(
        self,
        center: tuple[float, float, float],
        side: float,
        confirmed: bool,
    ) -> None:
        values = tuple(float(value) for value in center)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("refine cube center must be finite")
        if not math.isfinite(side) or side <= 0.0:
            raise ValueError("refine cube side must be finite and positive")
        self._refine_cube_center = values
        self._refine_cube_side = float(side)
        self._refine_cube_confirmed = bool(confirmed)
        self.invalidate()

    def bind_refine_cube_edit(
        self,
        handler: Callable[[tuple[float, float, float], float], None] | None,
    ) -> None:
        """Publish direct gizmo edits in viewport Z-up coordinates."""
        self._require_open()
        self._refine_cube_edit_handler = handler

    def set_refine_cube_edit_enabled(self, enabled: bool) -> None:
        self._refine_cube_edit_enabled = bool(enabled)
        if not self._refine_cube_edit_enabled:
            self._refine_gizmo_drag = None

    def clear_refine_cube(self) -> None:
        self._refine_cube_center = None
        self._refine_cube_side = 0.0
        self._refine_cube_confirmed = False
        self._refine_gizmo_drag = None
        self.invalidate()

    def begin_refine_cube_pick(
        self,
        handler: Callable[[tuple[float, float, float], float], None],
    ) -> None:
        self._require_open()
        if not self._mesh_items:
            raise ValueError("a mesh must be loaded before selecting a refine cube")
        self._refine_cube_pick_handler = handler
        self._refine_gizmo_drag = None

    def cancel_refine_cube_pick(self) -> None:
        self._refine_cube_pick_handler = None

    def _handle_pointer_press(self, x: float, y: float, button: int) -> bool:
        del button
        handler = self._refine_cube_pick_handler
        if handler is None:
            return self._begin_refine_gizmo_drag(x, y)
        width, height = self.surface.size
        ray = self._camera.screen_ray((x, y), width, height)
        if ray is None:
            return True
        origin = np.asarray(tuple(ray.origin), dtype=np.float32)
        direction = np.asarray(tuple(ray.direction), dtype=np.float32)
        nearest = None
        nearest_distance = math.inf
        for item in self._mesh_items:
            if item.positions is None or item.triangles is None:
                continue
            hit = _nearest_ray_triangle_hit(
                origin, direction, item.positions, item.triangles
            )
            if hit is None:
                continue
            distance = float(np.linalg.norm(hit - origin))
            if distance < nearest_distance:
                nearest = hit
                nearest_distance = distance
        if nearest is None:
            return True
        if self._model_bounds_min is None or self._model_bounds_max is None:
            side = 1.0
        else:
            extent = self._model_bounds_max - self._model_bounds_min
            side = max(float(np.max(extent)) * 0.2, 1e-4)
        center = nearest + direction * (side * 0.5)
        self._refine_cube_pick_handler = None
        handler(tuple(map(float, center)), side)
        return True

    def _gizmo_length(self) -> float:
        if self._model_bounds_min is None or self._model_bounds_max is None:
            model_length = self._refine_cube_side
        else:
            model_length = float(np.max(
                self._model_bounds_max - self._model_bounds_min
            ))
        return max(self._refine_cube_side * 0.50, model_length * 0.055, 1e-4)

    def _size_handle_position(self) -> np.ndarray:
        center = np.asarray(self._refine_cube_center, dtype=np.float64)
        return center + np.asarray((1.0, -1.0, 1.0)) * (
            self._refine_cube_side * 0.5
        )

    def _pick_refine_gizmo_handle(self, x: float, y: float) -> str | None:
        if self._refine_cube_center is None:
            return None
        width, height = self.surface.size
        center = np.asarray(self._refine_cube_center, dtype=np.float64)
        center_screen = self._camera.project_world_to_screen(
            center, width, height
        )
        if center_screen is None:
            return None
        pointer = np.asarray((x, y), dtype=np.float64)
        size_screen = self._camera.project_world_to_screen(
            self._size_handle_position(), width, height
        )
        if size_screen is not None:
            distance, parameter = _point_segment_distance(
                pointer, center_screen, size_screen
            )
            if (
                float(np.linalg.norm(pointer - size_screen)) <= 14.0
                or (distance <= 8.0 and parameter >= 0.45)
            ):
                return "size"
        length = self._gizmo_length()
        candidates = []
        for name, (direction, _color) in _REFINE_GIZMO_AXES.items():
            end_screen = self._camera.project_world_to_screen(
                center + direction * length, width, height
            )
            if end_screen is None:
                continue
            distance, parameter = _point_segment_distance(
                pointer, center_screen, end_screen
            )
            if distance <= 10.0 and parameter >= 0.12:
                candidates.append((distance, -parameter, name))
        return min(candidates)[2] if candidates else None

    def _begin_refine_gizmo_drag(self, x: float, y: float) -> bool:
        if (
            not self._refine_cube_edit_enabled
            or self._refine_cube_edit_handler is None
        ):
            return False
        kind = self._pick_refine_gizmo_handle(x, y)
        if kind is None:
            return False
        center = np.asarray(self._refine_cube_center, dtype=np.float64)
        direction = (
            _REFINE_SIZE_DIRECTION.copy()
            if kind == "size"
            else _REFINE_GIZMO_AXES[kind][0].copy()
        )
        direction /= np.linalg.norm(direction)
        view = np.asarray(self._camera.direction_from_target(), dtype=np.float64)
        plane_normal = view - direction * float(np.dot(view, direction))
        normal_length = float(np.linalg.norm(plane_normal))
        if normal_length <= 1e-6:
            helper = min(
                np.eye(3, dtype=np.float64),
                key=lambda candidate: abs(float(np.dot(candidate, direction))),
            )
            plane_normal = helper - direction * float(
                np.dot(helper, direction)
            )
            normal_length = float(np.linalg.norm(plane_normal))
        plane_normal /= normal_length
        width, height = self.surface.size
        ray = self._camera.screen_ray((x, y), width, height)
        if ray is None:
            return False
        hit = _ray_plane_intersection(
            np.asarray(tuple(ray.origin), dtype=np.float64),
            np.asarray(tuple(ray.direction), dtype=np.float64),
            center,
            plane_normal,
        )
        if hit is None:
            return False
        self._refine_gizmo_drag = _RefineGizmoDrag(
            kind=kind,
            original_center=center,
            original_side=self._refine_cube_side,
            constraint_direction=direction,
            plane_normal=plane_normal,
            start_parameter=float(np.dot(hit - center, direction)),
        )
        return True

    def _handle_gizmo_pointer_move(self, x: float, y: float) -> bool:
        drag = self._refine_gizmo_drag
        if drag is None:
            return False
        width, height = self.surface.size
        ray = self._camera.screen_ray((x, y), width, height)
        if ray is None:
            return True
        hit = _ray_plane_intersection(
            np.asarray(tuple(ray.origin), dtype=np.float64),
            np.asarray(tuple(ray.direction), dtype=np.float64),
            drag.original_center,
            drag.plane_normal,
        )
        if hit is None:
            return True
        parameter = float(np.dot(
            hit - drag.original_center, drag.constraint_direction
        ))
        delta = parameter - drag.start_parameter
        center = drag.original_center.copy()
        side = drag.original_side
        if drag.kind == "size":
            side = max(
                drag.original_side + delta * (2.0 / math.sqrt(3.0)),
                1e-4,
            )
        else:
            center += drag.constraint_direction * delta
        self.set_refine_cube(tuple(map(float, center)), side, False)
        handler = self._refine_cube_edit_handler
        if handler is not None:
            handler(tuple(map(float, center)), float(side))
        return True

    def _handle_gizmo_pointer_release(
        self, x: float, y: float, button: int
    ) -> bool:
        del x, y, button
        if self._refine_gizmo_drag is None:
            return False
        self._refine_gizmo_drag = None
        return True

    def set_shading_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in RECONSTRUCTION_SHADING_MODES:
            raise ValueError(f"unsupported reconstruction shading mode: {mode}")
        if normalized == self._shading_mode:
            return
        self._shading_mode = normalized
        self.invalidate()

    def set_point_cloud_color_mode(self, mode: str) -> None:
        """Switch uploaded point colors without changing point positions."""

        self._require_open()
        normalized = str(mode).strip().lower()
        if normalized not in POINT_CLOUD_COLOR_MODES:
            raise ValueError(f"unsupported point-cloud color mode: {mode}")
        if self._point_cloud_positions is None:
            raise ValueError("no point cloud is loaded")
        if (
                normalized == "confidence"
                and self._point_cloud_confidence_colors is None):
            raise ValueError("the point cloud has no confidence colors")
        if normalized == self._point_cloud_color_mode:
            return
        colors = (
            self._point_cloud_confidence_colors
            if normalized == "confidence"
            else self._point_cloud_image_colors
        )
        if colors is None or not self._point_cloud.upload_srgb(
                self._graphics.context, self._point_cloud_positions, colors):
            raise RuntimeError("Termin failed to update point-cloud colors")
        self._point_cloud_color_mode = normalized
        self._notify_point_color_state()
        self.invalidate()

    def light_from_camera(self) -> None:
        self._require_open()
        self._light_direction = self._camera.direction_from_target()
        self.invalidate()

    @property
    def refine_placement(self) -> ReconstructionRefinePlacement:
        return self._refine_placement

    def bind_refine_placement(
        self,
        placement: ReconstructionRefinePlacement,
        pivot: tuple[float, float, float],
        handler: Callable[[ReconstructionRefinePlacement], None] | None,
    ) -> None:
        """Bind a delta to only the editable mesh group in this viewport."""
        self._refine_placement_handler = handler
        self.preview_refine_placement(placement, pivot=pivot)

    def preview_refine_placement(
        self,
        placement: ReconstructionRefinePlacement,
        *,
        pivot: tuple[float, float, float] | None = None,
    ) -> None:
        self._require_open()
        if pivot is not None:
            self._refine_placement_pivot = tuple(map(float, pivot))
        self._refine_placement = placement
        self._model_transform = np.asarray(
            refine_placement_matrix(
                placement, self._refine_placement_pivot
            ),
            dtype=np.float32,
        )
        self.invalidate()

    def commit_refine_placement(
        self, placement: ReconstructionRefinePlacement
    ) -> None:
        self.preview_refine_placement(placement)
        if self._refine_placement_handler is not None:
            self._refine_placement_handler(placement)

    def clear_model(self) -> None:
        self._require_open()
        self._destroy_model_textures()
        self._mesh_items.clear()
        self._model_bounds_min = None
        self._model_bounds_max = None
        self.cancel_refine_cube_pick()
        self._refine_gizmo_drag = None
        self._point_cloud.release(self._graphics.context)
        self._clear_point_color_data()
        self._glb_documents.clear()
        self._model_transform = np.eye(4, dtype=np.float32)
        self.invalidate()

    def load_glb(
            self, path: str, *, fit_camera: bool | None = None
    ) -> tuple[int, int, int]:
        self._require_open()
        had_model = bool(self._mesh_items) or not self._point_cloud.empty
        source, mesh_items, vertices, indices, bounds_min, bounds_max = (
            self._read_glb_items(path, editable=True, mesh_offset=0)
        )
        if not mesh_items:
            raise RuntimeError("Generated GLB contains no meshes")
        self._destroy_model_textures()
        self._point_cloud.release(self._graphics.context)
        self._clear_point_color_data()
        self._glb_documents = [source]
        self._mesh_items = mesh_items
        self._model_bounds_min = bounds_min
        self._model_bounds_max = bounds_max
        if (
            bounds_min is not None
            and bounds_max is not None
            and _should_fit_camera(had_model, fit_camera)
        ):
            self._camera.fit(bounds_min, bounds_max)
            self._light_direction = self._camera.direction_from_target()
        self.invalidate()
        return vertices, indices // 3, len(mesh_items)

    def load_comparison_glbs(
        self,
        reference_path: str,
        editable_path: str,
        *,
        fit_camera: bool | None = None,
    ) -> tuple[int, int, int]:
        """Load a fixed base reference and a separately transformable fragment."""
        self._require_open()
        had_model = bool(self._mesh_items) or not self._point_cloud.empty
        (
            reference,
            reference_items,
            reference_vertices,
            reference_indices,
            reference_min,
            reference_max,
        ) = self._read_glb_items(
            reference_path,
            editable=False,
            mesh_offset=0,
            fallback_color=(0.34, 0.38, 0.44, 1.0),
        )
        (
            editable,
            editable_items,
            editable_vertices,
            editable_indices,
            editable_min,
            editable_max,
        ) = self._read_glb_items(
            editable_path,
            editable=True,
            mesh_offset=len(reference_items),
            fallback_color=(0.95, 0.55, 0.18, 1.0),
        )
        if not reference_items or not editable_items:
            raise RuntimeError("Refine comparison requires base and local meshes")
        self._destroy_model_textures()
        self._point_cloud.release(self._graphics.context)
        self._clear_point_color_data()
        self._glb_documents = [reference, editable]
        # Draw the proposal first.  With the normal depth test, coincident
        # base fragments then fail the equal-depth test instead of hiding the
        # orange local surface at its automatic registration pose.
        self._mesh_items = [*editable_items, *reference_items]
        bounds_min = np.minimum(reference_min, editable_min)
        bounds_max = np.maximum(reference_max, editable_max)
        self._model_bounds_min = bounds_min
        self._model_bounds_max = bounds_max
        if _should_fit_camera(had_model, fit_camera):
            self._camera.fit(bounds_min, bounds_max)
            self._light_direction = self._camera.direction_from_target()
        self.invalidate()
        return (
            reference_vertices + editable_vertices,
            (reference_indices + editable_indices) // 3,
            len(self._mesh_items),
        )

    def load_point_cloud(
            self, path: str, *, fit_camera: bool | None = None
    ) -> int:
        self._require_open()
        from ..native_ply import load_ply_points

        data = load_ply_points(path)
        return self.load_point_cloud_data(
            data.positions,
            data.colors,
            fit_camera=fit_camera,
        )

    def load_point_cloud_data(
            self,
            positions: np.ndarray,
            colors: np.ndarray,
            *,
            fit_camera: bool | None = None,
            confidence_colors: np.ndarray | None = None,
            color_mode: str = "image",
            confidence_legend: str = "",
    ) -> int:
        """Upload an in-memory sRGB point cloud without a file round-trip."""

        self._require_open()
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        if (
                positions.ndim != 2
                or positions.shape[1:] != (3,)
                or len(positions) == 0):
            raise ValueError("Point cloud positions must be a non-empty Nx3 array")
        if colors.shape != positions.shape:
            raise ValueError("Point cloud colors must match the Nx3 positions")
        if not np.isfinite(positions).all() or not np.isfinite(colors).all():
            raise ValueError("Point cloud data must be finite")
        colors = np.ascontiguousarray(np.clip(colors, 0.0, 1.0))
        normalized_mode = str(color_mode).strip().lower()
        if normalized_mode not in POINT_CLOUD_COLOR_MODES:
            raise ValueError(
                f"unsupported point-cloud color mode: {color_mode}")
        if confidence_colors is not None:
            confidence_colors = np.ascontiguousarray(
                confidence_colors, dtype=np.float32)
            if confidence_colors.shape != positions.shape:
                raise ValueError(
                    "Point cloud confidence colors must match Nx3 positions")
            if not np.isfinite(confidence_colors).all():
                raise ValueError("Point cloud confidence colors must be finite")
            confidence_colors = np.ascontiguousarray(
                np.clip(confidence_colors, 0.0, 1.0))
        elif normalized_mode == "confidence":
            raise ValueError("the point cloud has no confidence colors")
        upload_colors = (
            confidence_colors
            if normalized_mode == "confidence"
            else colors
        )
        had_model = bool(self._mesh_items) or not self._point_cloud.empty
        if not self._point_cloud.upload_srgb(
            self._graphics.context, positions, upload_colors
        ):
            raise RuntimeError("Termin failed to upload the reconstruction point cloud")
        self._point_cloud_positions = positions
        self._point_cloud_image_colors = colors
        self._point_cloud_confidence_colors = confidence_colors
        self._point_cloud_confidence_legend = str(confidence_legend)
        self._point_cloud_color_mode = normalized_mode
        self._destroy_model_textures()
        self._mesh_items.clear()
        self._model_bounds_min = None
        self._model_bounds_max = None
        self.cancel_refine_cube_pick()
        self._glb_documents.clear()
        if _should_fit_camera(had_model, fit_camera):
            self._camera.fit(positions.min(axis=0), positions.max(axis=0))
            self._light_direction = self._camera.direction_from_target()
        self._notify_point_color_state()
        self.invalidate()
        return len(positions)

    def _clear_point_color_data(self) -> None:
        self._point_cloud_positions = None
        self._point_cloud_image_colors = None
        self._point_cloud_confidence_colors = None
        self._point_cloud_confidence_legend = ""
        self._point_cloud_color_mode = "image"
        self._notify_point_color_state()

    def _notify_point_color_state(self) -> None:
        if self._point_color_state_changed is not None:
            self._point_color_state_changed(
                self.point_cloud_has_confidence,
                self._point_cloud_color_mode,
                self._point_cloud_confidence_legend,
            )

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
            item_mvp = (
                mvp @ self._model_transform
                if item.editable
                else mvp
            )
            if self._shading_mode == "wireframe":
                if item.wire_mesh is None:
                    continue
                ctx.bind_shader(self._vertex_shader, self._fragment_shader)
                self._set_draw_state(
                    item_mvp,
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
                item_mvp,
                item.color,
                self._light_direction,
                display_mode,
            )
            draw_tc_mesh(
                ctx,
                item.smooth_mesh
                if use_smooth_normals and item.smooth_mesh is not None
                else item.mesh,
            )
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
        if self._refine_cube_center is not None:
            cube_transform = np.eye(4, dtype=np.float32)
            cube_transform[:3, :3] *= self._refine_cube_side
            cube_transform[:3, 3] = self._refine_cube_center
            cube_mvp = mvp @ cube_transform
            color = (
                (0.20, 0.95, 0.42, 0.16)
                if self._refine_cube_confirmed
                else (1.00, 0.55, 0.10, 0.18)
            )
            wire_color = (
                (0.30, 1.00, 0.50, 1.0)
                if self._refine_cube_confirmed
                else (1.00, 0.62, 0.16, 1.0)
            )
            ctx.bind_shader(self._vertex_shader, self._fragment_shader)
            ctx.set_depth_test(False)
            ctx.set_depth_write(False)
            ctx.set_blend(True)
            self._set_draw_state(
                cube_mvp, color, self._light_direction,
                _SHADING_MODE_IDS["unlit"],
            )
            draw_tc_mesh(ctx, self._refine_cube_fill_mesh)
            ctx.set_blend(False)
            self._set_draw_state(
                cube_mvp, wire_color, self._light_direction,
                _SHADING_MODE_IDS["unlit"],
            )
            draw_tc_mesh(ctx, self._refine_cube_wire_mesh)
            center = np.asarray(self._refine_cube_center, dtype=np.float64)
            gizmo_length = self._gizmo_length()
            for _name, (direction, axis_color) in _REFINE_GIZMO_AXES.items():
                arrow_transform = _oriented_transform(
                    center, direction, gizmo_length
                )
                self._set_draw_state(
                    mvp @ arrow_transform,
                    axis_color,
                    self._light_direction,
                    _SHADING_MODE_IDS["unlit"],
                )
                draw_tc_mesh(ctx, self._refine_gizmo_arrow_mesh)
            size_length = self._refine_cube_side * math.sqrt(3.0) * 0.5
            size_transform = _oriented_transform(
                center, _REFINE_SIZE_DIRECTION, size_length
            )
            self._set_draw_state(
                mvp @ size_transform,
                (0.95, 0.25, 0.92, 1.0),
                self._light_direction,
                _SHADING_MODE_IDS["unlit"],
            )
            draw_tc_mesh(ctx, self._refine_gizmo_arrow_mesh)
            handle_transform = np.eye(4, dtype=np.float32)
            handle_size = gizmo_length * 0.10
            handle_transform[:3, :3] *= handle_size
            handle_transform[:3, 3] = self._size_handle_position()
            self._set_draw_state(
                mvp @ handle_transform,
                (1.0, 0.72, 1.0, 1.0),
                self._light_direction,
                _SHADING_MODE_IDS["unlit"],
            )
            draw_tc_mesh(ctx, self._refine_cube_fill_mesh)
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
        self._model_bounds_min = None
        self._model_bounds_max = None
        self._glb_documents.clear()

    def _read_glb_items(
        self,
        path: str,
        *,
        editable: bool,
        mesh_offset: int,
        fallback_color: tuple[float, float, float, float] = (
            0.34, 0.72, 0.95, 1.0
        ),
    ):
        from ..native_glb import NativeGLBDocument

        source = NativeGLBDocument(path)
        mesh_items = []
        vertices = 0
        indices = 0
        bounds_min = None
        bounds_max = None
        for index, info in enumerate(source.meshes):
            if info.skinned:
                raise RuntimeError(
                    "The reconstruction viewport supports static GLB meshes only"
                )
            mesh = source.build_mesh(
                index,
                _mesh_resource_id(
                    self._resource_namespace, mesh_offset + index
                ),
                name=info.name or f"Reconstruction mesh {index}",
                convert_to_z_up=True,
            )
            texture = None
            color = fallback_color
            base_color = source.base_color_texture(index)
            if base_color is not None:
                width, height, pixels = _decode_texture(base_color.payload)
                texture = self._graphics.create_texture_rgba8(
                    width, height, pixels, TextureEncoding.SRGB
                )
                color = base_color.factor
            authored_normals = mesh.vertex_normals is not None
            smooth_mesh = None
            if not authored_normals and texture is None:
                smooth_mesh = _build_smooth_mesh(mesh)
            mesh_items.append(_MeshRenderItem(
                mesh,
                texture,
                color,
                has_normals=authored_normals or smooth_mesh is not None,
                smooth_mesh=smooth_mesh,
                editable=editable,
                positions=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                triangles=(
                    np.ascontiguousarray(mesh.triangles, dtype=np.uint32)
                    if mesh.triangles is not None
                    else None
                ),
            ))
            vertices += int(mesh.vertex_count)
            indices += int(mesh.index_count)
            payload = np.asarray(
                mesh.mesh.get_vertices_buffer(), dtype=np.float32
            )
            stride = int(mesh.stride) // 4
            if stride >= 3 and payload.size >= stride:
                positions = payload.reshape(-1, stride)[:, :3]
                local_min = positions.min(axis=0)
                local_max = positions.max(axis=0)
                bounds_min = (
                    local_min
                    if bounds_min is None
                    else np.minimum(bounds_min, local_min)
                )
                bounds_max = (
                    local_max
                    if bounds_max is None
                    else np.maximum(bounds_max, local_max)
                )
        return (
            source, mesh_items, vertices, indices, bounds_min, bounds_max
        )

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
