from __future__ import annotations

from diffusion_editor import native_glb


def test_sdk_glb_native_boundary_loads_cgltf_without_high_level_package():
    module = native_glb._load_native_module()

    assert module.backend_info()["name"] == "cgltf"
