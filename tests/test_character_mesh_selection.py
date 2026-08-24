import pytest

from diffusion_editor.training.character_mesh_selection import (
    CharacterMeshCandidate,
    select_character_mesh_names,
)


def test_only_meshes_are_an_exact_allow_list():
    candidates = [
        CharacterMeshCandidate("body-a"),
        CharacterMeshCandidate("body-b"),
        CharacterMeshCandidate("hidden-hair", render_hidden=True),
    ]

    assert select_character_mesh_names(
        candidates,
        mode="opened-blend",
        only_meshes=["body-b", "hidden-hair"],
    ) == ["body-b", "hidden-hair"]


def test_only_meshes_reject_missing_names():
    candidates = [CharacterMeshCandidate("body")]

    with pytest.raises(
        ValueError, match="requested character meshes are missing: missing"
    ):
        select_character_mesh_names(
            candidates,
            mode="opened-blend",
            only_meshes=["missing"],
        )


def test_include_meshes_remain_additive_for_opened_blends():
    candidates = [
        CharacterMeshCandidate("body"),
        CharacterMeshCandidate("authored-hidden", render_hidden=True),
        CharacterMeshCandidate("WGT-control"),
    ]

    assert select_character_mesh_names(
        candidates,
        mode="opened-blend",
        include_meshes=["authored-hidden"],
    ) == ["body", "authored-hidden"]
