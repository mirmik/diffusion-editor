from __future__ import annotations

import json
import pytest

from diffusion_editor.generation.reconstruction_workspace import (
    PIXAL3D_PIPELINE,
    PIXAL3D_PRESENTED_OPERATION_KEYS,
    ReconstructionWorkspace,
    WorkspaceOperationStatus,
    WorkspacePreviewKind,
    build_legacy_workspace,
    pixal3d_presented_operations,
)
from diffusion_editor.generation.types import (
    RECONSTRUCTION_STAGES,
    ReconstructionParameters,
    ReconstructionLrVariant,
    ReconstructionRun,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageStatus,
)


def _ready_source(
        workspace: ReconstructionWorkspace,
        *,
        operation_id: str = "source-op",
        artifact_id: str = "source-artifact",
        content_hash: str = "source-hash",
):
    operation = workspace.plan_operation(
        "source.prepare",
        parameters={"background": "removed"},
        model_identity="pixal3d:test",
        worker_protocol="1",
        operation_id=operation_id,
    )
    artifact = workspace.publish_artifact(
        operation.operation_id,
        "source",
        artifact_id=artifact_id,
        content_hash=content_hash,
        path="source.png",
    )
    workspace.set_operation_status(operation.operation_id,
                                   WorkspaceOperationStatus.READY)
    return operation, artifact


def test_pixal_pipeline_exposes_lr_and_local_detail_checkpoints() -> None:
    keys = {operation.key for operation in PIXAL3D_PIPELINE.operations}
    assert {
        "lr.refine",
        "local.upscale",
        "local.generate_geometry",
        "local.register",
        "local.fuse",
        "texture.local_generate",
        "texture.transfer",
    } <= keys
    local_outputs = {
        output.role
        for operation in PIXAL3D_PIPELINE.operations_in_group("local")
        for output in operation.outputs
    }
    assert {
        "upscaled_condition",
        "local_mesh",
        "registration_overlay",
        "overlap_preview",
        "fused_mesh",
    } <= local_outputs


def test_presented_pipeline_contains_only_current_decision_points() -> None:
    assert PIXAL3D_PRESENTED_OPERATION_KEYS == {
        "source.prepare",
        "sparse.generate",
        "lr.generate",
        "hr.coordinates",
        "hr.generate",
        "hr.refine",
        "texture.generate",
        "texture.refine",
        "final.assemble",
    }
    assert pixal3d_presented_operations("geometry") == ()
    assert pixal3d_presented_operations("local") == ()


def test_fingerprint_is_stable_and_uses_inputs_parameters_and_protocol() -> None:
    left = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    right = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    _, left_source = _ready_source(left)
    _, right_source = _ready_source(right)

    first = left.plan_operation(
        "sparse.generate",
        input_artifact_ids=(left_source.artifact_id,),
        parameters={"steps": 12, "seed": 3},
        model_identity="pixal3d:abc",
        worker_protocol="2",
    )
    same = right.plan_operation(
        "sparse.generate",
        input_artifact_ids=(right_source.artifact_id,),
        parameters={"seed": 3, "steps": 12},
        model_identity="pixal3d:abc",
        worker_protocol="2",
    )
    changed = left.plan_operation(
        "sparse.generate",
        input_artifact_ids=(left_source.artifact_id,),
        parameters={"steps": 13, "seed": 3},
        model_identity="pixal3d:abc",
        worker_protocol="2",
    )
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_preview_selection_does_not_change_accepted_variant() -> None:
    workspace = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    source_operation, source = _ready_source(workspace)
    workspace.accept_operation(source_operation.operation_id)

    sparse = workspace.plan_operation(
        "sparse.generate",
        input_artifact_ids=(source.artifact_id,),
        parameters={"seed": 1},
        model_identity="pixal3d:test",
        worker_protocol="1",
        variant_label="Sparse A",
    )
    preview = workspace.publish_artifact(
        sparse.operation_id,
        "sparse_preview",
        artifact_id="sparse-preview",
        content_hash="preview-hash",
    )
    workspace.set_operation_status(
        sparse.operation_id, WorkspaceOperationStatus.READY
    )
    workspace.select_artifact(preview.artifact_id)
    assert workspace.accepted_operations == {"source": source_operation.operation_id}

    workspace.accept_operation(sparse.operation_id)
    assert workspace.selected_artifact_id == preview.artifact_id
    assert workspace.accepted_operations["sparse"] == sparse.operation_id


def test_only_ready_or_cached_approval_points_can_be_accepted() -> None:
    workspace = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    source, _artifact = _ready_source(workspace)
    workspace.set_operation_status(source.operation_id,
                                   WorkspaceOperationStatus.PLANNED)
    with pytest.raises(ValueError, match="ready or cached"):
        workspace.accept_operation(source.operation_id)


def test_operation_rejects_unknown_inputs_and_output_roles() -> None:
    workspace = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    with pytest.raises(ValueError, match="inputs are missing"):
        workspace.plan_operation(
            "sparse.generate",
            input_artifact_ids=("missing",),
            model_identity="pixal3d:test",
            worker_protocol="1",
        )
    operation, _artifact = _ready_source(workspace)
    with pytest.raises(ValueError, match="does not publish role"):
        workspace.publish_artifact(operation.operation_id, "not-an-output")


def test_legacy_projection_maps_ready_stages_and_preview_artifacts() -> None:
    statuses = {
        stage: ReconstructionStageStatus.PENDING
        for stage in RECONSTRUCTION_STAGES
    }
    statuses[ReconstructionStage.SOURCE_IMAGE] = ReconstructionStageStatus.READY
    statuses[ReconstructionStage.SPARSE_OCCUPANCY] = (
        ReconstructionStageStatus.READY
    )
    statuses[ReconstructionStage.LR_SHAPE_FLOW] = (
        ReconstructionStageStatus.RUNNING
    )
    artifacts = {
        ReconstructionStage.SOURCE_IMAGE: ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE, "/tmp/source.png", "image"
        ),
        ReconstructionStage.SPARSE_OCCUPANCY: ReconstructionStageArtifact(
            ReconstructionStage.SPARSE_OCCUPANCY, "/tmp/sparse.glb", "mesh"
        ),
    }
    workspace = build_legacy_workspace(
        ReconstructionParameters(), statuses, artifacts, (), None
    )
    assert workspace is not None
    assert workspace.operations_for_spec("source.prepare")[0].status is (
        WorkspaceOperationStatus.READY
    )
    assert workspace.operations_for_spec("sparse.generate")[0].status is (
        WorkspaceOperationStatus.READY
    )
    assert workspace.operations_for_spec("lr.generate")[0].status is (
        WorkspaceOperationStatus.RUNNING
    )
    sparse_artifact = workspace.artifacts_for_operation(
        workspace.operations_for_spec("sparse.generate")[0].operation_id
    )[0]
    assert sparse_artifact.path == "/tmp/sparse.glb"
    assert sparse_artifact.preview_kind is WorkspacePreviewKind.MESH


def test_legacy_projection_fingerprints_only_operation_parameters() -> None:
    statuses = {
        stage: ReconstructionStageStatus.PENDING
        for stage in RECONSTRUCTION_STAGES
    }
    original = build_legacy_workspace(
        ReconstructionParameters(pixal3d_lr_seed=20),
        statuses, {}, (), None,
    )
    changed = build_legacy_workspace(
        ReconstructionParameters(pixal3d_lr_seed=21),
        statuses, {}, (), None,
    )

    assert original.operations_for_spec("sparse.generate")[0].fingerprint == (
        changed.operations_for_spec("sparse.generate")[0].fingerprint
    )
    assert original.operations_for_spec("lr.generate")[0].fingerprint != (
        changed.operations_for_spec("lr.generate")[0].fingerprint
    )


def test_legacy_projection_exposes_refine_runs_as_stage_local_variants() -> None:
    ready_statuses = tuple(
        ReconstructionStageStatus.READY for _stage in RECONSTRUCTION_STAGES
    )
    base = ReconstructionRun(
        "base", ReconstructionRunKind.BASE,
        "/tmp/base.glb", "/tmp/source.png",
        checkpoint_path="/tmp/base.pt",
        stage_statuses=ready_statuses,
        stage_artifacts=(ReconstructionStageArtifact(
            ReconstructionStage.HR_SHAPE_LATENT,
            "/tmp/base-hr.glb",
            "mesh",
        ),),
    )
    refined = ReconstructionRun(
        "refined", ReconstructionRunKind.MASKED_REFINE,
        "/tmp/refined.glb", "/tmp/source.png",
        checkpoint_path="/tmp/refined.pt",
        parent_run_id="base",
        stage_statuses=ready_statuses,
        stage_artifacts=(ReconstructionStageArtifact(
            ReconstructionStage.HR_SHAPE_LATENT,
            "/tmp/refined-hr.glb",
            "mesh",
        ),),
        refine_generated_path="/tmp/refined-generated.glb",
    )
    workspace = build_legacy_workspace(
        ReconstructionParameters(), {}, {}, (base, refined), "refined"
    )
    assert workspace is not None
    assert [
        operation.variant_label
        for operation in workspace.operations_for_spec("hr.generate")
    ] == ["Base"]
    assert [
        operation.variant_label
        for operation in workspace.operations_for_spec("hr.refine")
    ] == ["Geometry refine 1"]
    assert workspace.selected_artifact_id is not None
    selected = workspace.artifact(workspace.selected_artifact_id)
    assert selected.path in {"/tmp/refined-hr.glb", "/tmp/refined.glb"}
    refined_operation = workspace.operations_for_spec("hr.refine")[0]
    generated = next(
        item for item in workspace.artifacts_for_operation(
            refined_operation.operation_id
        )
        if item.role == "refine_generated_mesh"
    )
    assert generated.path == "/tmp/refined-generated.glb"


def test_legacy_projection_keeps_base_and_refined_lr_previews_separate() -> None:
    base_preview = ReconstructionStageArtifact(
        ReconstructionStage.LR_SHAPE_LATENT,
        "/tmp/base-lr.glb",
        "mesh",
    )
    refined_preview = ReconstructionStageArtifact(
        ReconstructionStage.LR_SHAPE_LATENT,
        "/tmp/refined-lr.glb",
        "mesh",
    )
    base = ReconstructionLrVariant(
        "lr-base", "Base LR", "/tmp/base-lr.npz", "/tmp/source.png",
        base_preview,
    )
    refined = ReconstructionLrVariant(
        "lr-refined-1", "Refined LR 1", "/tmp/refined-lr.npz",
        "/tmp/source.png", refined_preview, parent_variant_id="lr-base",
        refine_generated_path="/tmp/lr-generated.glb",
    )
    statuses = {
        stage: ReconstructionStageStatus.READY
        for stage in RECONSTRUCTION_STAGES
    }
    workspace = build_legacy_workspace(
        ReconstructionParameters(),
        statuses,
        {ReconstructionStage.LR_SHAPE_LATENT: refined_preview},
        (),
        None,
        lr_variants=(base, refined),
        selected_lr_refine_source_id="lr-base",
    )

    generated = workspace.operations_for_spec("lr.generate")
    assert len(generated) == 1
    generated_artifacts = workspace.artifacts_for_operation(
        generated[0].operation_id
    )
    assert next(
        item.path for item in generated_artifacts if item.role == "lr_preview"
    ) == "/tmp/base-lr.glb"
    base_checkpoint = next(
        item for item in generated_artifacts if item.role == "lr_checkpoint"
    )
    assert json.loads(base_checkpoint.metadata_json)[
        "selected_refine_source"
    ] is True

    refined_operations = workspace.operations_for_spec("lr.refine")
    assert [item.variant_label for item in refined_operations] == [
        "Refined LR 1"
    ]
    refined_artifacts = workspace.artifacts_for_operation(
        refined_operations[0].operation_id
    )
    assert next(
        item.path for item in refined_artifacts if item.role == "lr_preview"
    ) == "/tmp/refined-lr.glb"
    assert next(
        item.path for item in refined_artifacts
        if item.role == "refine_generated_mesh"
    ) == "/tmp/lr-generated.glb"


def test_legacy_projection_is_pixal_only() -> None:
    parameters = ReconstructionParameters(backend="trellis2")
    assert build_legacy_workspace(parameters, {}, {}, (), None) is None
