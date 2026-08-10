from __future__ import annotations

from diffusion_editor import native_glb


def test_sdk_glb_native_boundary_loads_cgltf_without_high_level_package():
    module = native_glb._load_native_module()

    assert module.backend_info()["name"] == "cgltf"


class _FakeDocument:
    def primitive_info(self, mesh_index, primitive_index):
        assert (mesh_index, primitive_index) == (0, 0)
        return {"has_material": True, "material_index": 4}

    def material_info(self, material_index):
        assert material_index == 4
        return {
            "base_color_factor": (0.5, 0.75, 1.0, 1.0),
            "base_color_texture": {
                "present": True,
                "texture_index": 2,
                "texcoord": 0,
                "has_transform": False,
            },
        }

    def texture_info(self, texture_index):
        assert texture_index == 2
        return {"has_image": True, "image_index": 7}

    def image_info(self, image_index):
        assert image_index == 7
        return {"mime_type": "image/webp"}

    def image_payload(self, image_index):
        assert image_index == 7
        return b"encoded image"


def test_base_color_texture_resolves_embedded_image() -> None:
    document = native_glb.NativeGLBDocument.__new__(native_glb.NativeGLBDocument)
    document.meshes = (
        native_glb.NativeMeshInfo("mesh", 1, 3, 3, False),
    )
    document._document = _FakeDocument()

    texture = document.base_color_texture(0)

    assert texture == native_glb.NativeBaseColorTexture(
        payload=b"encoded image",
        mime_type="image/webp",
        factor=(0.5, 0.75, 1.0, 1.0),
    )


def test_base_color_texture_rejects_multi_primitive_mesh() -> None:
    document = native_glb.NativeGLBDocument.__new__(native_glb.NativeGLBDocument)
    document.meshes = (
        native_glb.NativeMeshInfo("mesh", 2, 6, 6, False),
    )
    document._document = _FakeDocument()

    assert document.base_color_texture(0) is None
