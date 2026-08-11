"""Immutable operation/artifact graph for experimental reconstruction work."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Mapping, Sequence
import uuid

from .types import (
    RECONSTRUCTION_STAGES,
    ReconstructionBackend,
    ReconstructionParameters,
    ReconstructionLrVariant,
    ReconstructionRun,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageStatus,
)


class WorkspaceArtifactKind(str, Enum):
    IMAGE = "image"
    MASK = "mask"
    VOLUME = "volume"
    POINT_CLOUD = "point_cloud"
    LATENT = "latent"
    CHECKPOINT = "checkpoint"
    MESH = "mesh"
    TEXTURE = "texture"
    METADATA = "metadata"


class WorkspacePreviewKind(str, Enum):
    NONE = "none"
    IMAGE = "image"
    MESH = "mesh"
    POINTS = "points"
    VOLUME = "volume"
    OVERLAY = "overlay"


class WorkspaceOperationStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    READY = "ready"
    CACHED = "cached"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactPortSpec:
    role: str
    label: str
    kind: WorkspaceArtifactKind
    preview_kind: WorkspacePreviewKind = WorkspacePreviewKind.NONE


@dataclass(frozen=True)
class PipelineGroupSpec:
    key: str
    label: str


@dataclass(frozen=True)
class PipelineOperationSpec:
    key: str
    label: str
    group_key: str
    input_roles: tuple[str, ...]
    outputs: tuple[ArtifactPortSpec, ...]
    acceptance_key: str | None = None
    description: str = ""


@dataclass(frozen=True)
class BackendPipelineDefinition:
    backend: ReconstructionBackend
    groups: tuple[PipelineGroupSpec, ...]
    operations: tuple[PipelineOperationSpec, ...]

    def __post_init__(self) -> None:
        group_keys = [group.key for group in self.groups]
        operation_keys = [operation.key for operation in self.operations]
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("pipeline group keys must be unique")
        if len(operation_keys) != len(set(operation_keys)):
            raise ValueError("pipeline operation keys must be unique")
        unknown = {
            operation.group_key for operation in self.operations
            if operation.group_key not in group_keys
        }
        if unknown:
            raise ValueError(f"operations reference unknown groups: {unknown}")

    def operation(self, key: str) -> PipelineOperationSpec:
        for operation in self.operations:
            if operation.key == key:
                return operation
        raise KeyError(key)

    def operations_in_group(
            self, group_key: str) -> tuple[PipelineOperationSpec, ...]:
        return tuple(
            operation for operation in self.operations
            if operation.group_key == group_key
        )


@dataclass(frozen=True)
class WorkspaceArtifact:
    artifact_id: str
    role: str
    kind: WorkspaceArtifactKind
    producer_operation_id: str
    path: str = ""
    content_hash: str = ""
    preview_kind: WorkspacePreviewKind = WorkspacePreviewKind.NONE
    metadata_json: str = "{}"


@dataclass(frozen=True)
class WorkspaceOperation:
    operation_id: str
    spec_key: str
    variant_label: str
    input_artifact_ids: tuple[str, ...]
    parameters_json: str
    model_identity: str
    worker_protocol: str
    fingerprint: str
    status: WorkspaceOperationStatus = WorkspaceOperationStatus.PLANNED
    output_artifact_ids: tuple[str, ...] = ()
    error: str = ""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ReconstructionWorkspace:
    """Session-local graph state kept separate from legacy reconstruction runs."""

    def __init__(self, definition: BackendPipelineDefinition) -> None:
        self.definition = definition
        self._operations: dict[str, WorkspaceOperation] = {}
        self._artifacts: dict[str, WorkspaceArtifact] = {}
        self._accepted_operations: dict[str, str] = {}
        self.selected_artifact_id: str | None = None

    @property
    def operations(self) -> tuple[WorkspaceOperation, ...]:
        return tuple(self._operations.values())

    @property
    def artifacts(self) -> tuple[WorkspaceArtifact, ...]:
        return tuple(self._artifacts.values())

    @property
    def accepted_operations(self) -> Mapping[str, str]:
        return dict(self._accepted_operations)

    def operation(self, operation_id: str) -> WorkspaceOperation:
        return self._operations[operation_id]

    def artifact(self, artifact_id: str) -> WorkspaceArtifact:
        return self._artifacts[artifact_id]

    def operations_for_spec(self, spec_key: str) -> tuple[WorkspaceOperation, ...]:
        return tuple(
            operation for operation in self._operations.values()
            if operation.spec_key == spec_key
        )

    def artifacts_for_operation(
            self, operation_id: str) -> tuple[WorkspaceArtifact, ...]:
        operation = self.operation(operation_id)
        return tuple(
            self._artifacts[artifact_id]
            for artifact_id in operation.output_artifact_ids
        )

    def plan_operation(
            self,
            spec_key: str,
            *,
            input_artifact_ids: Sequence[str] = (),
            parameters: Mapping[str, object] | None = None,
            model_identity: str,
            worker_protocol: str,
            variant_label: str = "Base",
            operation_id: str | None = None,
    ) -> WorkspaceOperation:
        self.definition.operation(spec_key)
        input_ids = tuple(map(str, input_artifact_ids))
        missing = [key for key in input_ids if key not in self._artifacts]
        if missing:
            raise ValueError(f"operation inputs are missing: {missing}")
        parameters_json = _canonical_json(dict(parameters or {}))
        input_identities = [
            self._artifacts[key].content_hash or f"artifact:{key}"
            for key in input_ids
        ]
        fingerprint_payload = {
            "backend": self.definition.backend.value,
            "operation": spec_key,
            "inputs": input_identities,
            "parameters": json.loads(parameters_json),
            "model": str(model_identity),
            "worker_protocol": str(worker_protocol),
        }
        fingerprint = hashlib.sha256(
            _canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        record = WorkspaceOperation(
            operation_id=operation_id or uuid.uuid4().hex,
            spec_key=spec_key,
            variant_label=str(variant_label),
            input_artifact_ids=input_ids,
            parameters_json=parameters_json,
            model_identity=str(model_identity),
            worker_protocol=str(worker_protocol),
            fingerprint=fingerprint,
        )
        if record.operation_id in self._operations:
            raise ValueError(f"duplicate operation id: {record.operation_id}")
        self._operations[record.operation_id] = record
        return record

    def set_operation_status(
            self,
            operation_id: str,
            status: WorkspaceOperationStatus,
            *,
            error: str = "",
    ) -> WorkspaceOperation:
        record = self.operation(operation_id)
        if not isinstance(status, WorkspaceOperationStatus):
            status = WorkspaceOperationStatus(status)
        updated = replace(record, status=status, error=str(error))
        self._operations[operation_id] = updated
        return updated

    def publish_artifact(
            self,
            operation_id: str,
            role: str,
            *,
            path: str = "",
            content_hash: str = "",
            kind: WorkspaceArtifactKind | None = None,
            preview_kind: WorkspacePreviewKind | None = None,
            metadata: Mapping[str, object] | None = None,
            artifact_id: str | None = None,
    ) -> WorkspaceArtifact:
        operation = self.operation(operation_id)
        spec = self.definition.operation(operation.spec_key)
        try:
            port = next(output for output in spec.outputs if output.role == role)
        except StopIteration as exc:
            raise ValueError(
                f"operation {spec.key!r} does not publish role {role!r}"
            ) from exc
        artifact = WorkspaceArtifact(
            artifact_id=artifact_id or uuid.uuid4().hex,
            role=role,
            kind=kind or port.kind,
            producer_operation_id=operation_id,
            path=str(path),
            content_hash=str(content_hash),
            preview_kind=preview_kind or port.preview_kind,
            metadata_json=_canonical_json(dict(metadata or {})),
        )
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"duplicate artifact id: {artifact.artifact_id}")
        self._artifacts[artifact.artifact_id] = artifact
        self._operations[operation_id] = replace(
            operation,
            output_artifact_ids=(
                *operation.output_artifact_ids, artifact.artifact_id
            ),
        )
        return artifact

    def select_artifact(self, artifact_id: str | None) -> None:
        if artifact_id is not None and artifact_id not in self._artifacts:
            raise KeyError(artifact_id)
        self.selected_artifact_id = artifact_id

    def accept_operation(self, operation_id: str) -> None:
        operation = self.operation(operation_id)
        if operation.status not in {
            WorkspaceOperationStatus.READY,
            WorkspaceOperationStatus.CACHED,
        }:
            raise ValueError("only ready or cached operations can be accepted")
        spec = self.definition.operation(operation.spec_key)
        if not spec.acceptance_key:
            raise ValueError(f"operation {spec.key!r} is not an approval point")
        self._accepted_operations[spec.acceptance_key] = operation_id


def _port(
        role: str,
        label: str,
        kind: WorkspaceArtifactKind,
        preview: WorkspacePreviewKind = WorkspacePreviewKind.NONE,
) -> ArtifactPortSpec:
    return ArtifactPortSpec(role, label, kind, preview)


PIXAL3D_PIPELINE = BackendPipelineDefinition(
    backend=ReconstructionBackend.PIXAL3D,
    groups=(
        PipelineGroupSpec("source", "Source preparation"),
        PipelineGroupSpec("sparse", "Sparse occupancy"),
        PipelineGroupSpec("lr", "LR shape"),
        PipelineGroupSpec("hr", "HR shape"),
        PipelineGroupSpec("local", "Local 1536 detail"),
        PipelineGroupSpec("geometry", "Geometry processing"),
        PipelineGroupSpec("texture", "Texture"),
        PipelineGroupSpec("final", "Final assembly"),
    ),
    operations=(
        PipelineOperationSpec(
            "source.prepare", "Prepare source", "source", (),
            (
                _port("source", "Conditioning image", WorkspaceArtifactKind.IMAGE,
                      WorkspacePreviewKind.IMAGE),
                _port("foreground_mask", "Foreground mask", WorkspaceArtifactKind.MASK,
                      WorkspacePreviewKind.IMAGE),
            ),
            "source", "Snapshot and normalize the source image and foreground mask.",
        ),
        PipelineOperationSpec(
            "sparse.generate", "Generate sparse occupancy", "sparse", ("source",),
            (
                _port("sparse_checkpoint", "Sparse checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("sparse_preview", "Sparse occupancy preview",
                      WorkspaceArtifactKind.VOLUME, WorkspacePreviewKind.VOLUME),
            ),
            "sparse", "Generate a coarse spatial support variant.",
        ),
        PipelineOperationSpec(
            "sparse.refine", "Refine sparse occupancy", "sparse",
            ("source", "sparse_checkpoint"),
            (
                _port("sparse_checkpoint", "Refined sparse checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("sparse_preview", "Refined occupancy preview",
                      WorkspaceArtifactKind.VOLUME, WorkspacePreviewKind.VOLUME),
            ),
            "sparse", "Optional backend-supported sparse variant refinement.",
        ),
        PipelineOperationSpec(
            "lr.generate", "Generate LR shape", "lr",
            ("source", "sparse_checkpoint"),
            (
                _port("lr_checkpoint", "LR shape checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("lr_latent", "LR shape latent", WorkspaceArtifactKind.LATENT),
                _port("lr_preview", "Decoded LR preview", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
            ),
            "lr_shape", "Generate the coarse LR shape used by HR support.",
        ),
        PipelineOperationSpec(
            "lr.refine", "Refine LR shape", "lr",
            ("source", "lr_checkpoint"),
            (
                _port("lr_checkpoint", "Refined LR checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("lr_latent", "Refined LR latent", WorkspaceArtifactKind.LATENT),
                _port("lr_preview", "Decoded refined LR preview",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
                _port("lr_mask_projection", "LR mask projection",
                      WorkspaceArtifactKind.MASK, WorkspacePreviewKind.IMAGE),
            ),
            "lr_shape", "Iterate coarse form before accepting it for HR upsampling.",
        ),
        PipelineOperationSpec(
            "hr.coordinates", "Upsample LR to HR coordinates", "hr",
            ("lr_checkpoint",),
            (
                _port("hr_coordinates", "HR coordinate checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("hr_coordinate_preview", "HR coordinate preview",
                      WorkspaceArtifactKind.POINT_CLOUD, WorkspacePreviewKind.POINTS),
            ),
            "hr_coordinates", "Derive HR support from the accepted LR variant.",
        ),
        PipelineOperationSpec(
            "hr.generate", "Generate HR shape", "hr",
            ("source", "hr_coordinates"),
            (
                _port("hr_checkpoint", "HR shape checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("hr_latent", "HR shape latent", WorkspaceArtifactKind.LATENT),
                _port("hr_mesh", "Decoded HR mesh", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
            ),
            "hr_shape", "Generate high-resolution shape on the accepted support.",
        ),
        PipelineOperationSpec(
            "hr.refine", "Refine HR shape", "hr",
            ("source", "hr_checkpoint"),
            (
                _port("hr_checkpoint", "Refined HR checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("hr_mesh", "Refined HR mesh", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
                _port("hr_mask_projection", "HR mask projection",
                      WorkspaceArtifactKind.MASK, WorkspacePreviewKind.IMAGE),
            ),
            "hr_shape", "Create a global or masked HR variant.",
        ),
        PipelineOperationSpec(
            "local.prepare_roi", "Prepare mask and 3D ROI", "local",
            ("source", "hr_mesh"),
            (
                _port("refine_mask", "Refine mask", WorkspaceArtifactKind.MASK,
                      WorkspacePreviewKind.IMAGE),
                _port("source_crop", "Source crop", WorkspaceArtifactKind.IMAGE,
                      WorkspacePreviewKind.IMAGE),
                _port("roi", "ROI transform", WorkspaceArtifactKind.METADATA),
                _port("roi_overlay", "ROI cube over base mesh",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.OVERLAY),
            ),
            "local_roi", "Project the mask, crop the source and define local-to-global scale.",
        ),
        PipelineOperationSpec(
            "local.upscale", "Upscale local conditioning", "local",
            ("source_crop",),
            (
                _port("resized_condition", "Resized condition",
                      WorkspaceArtifactKind.IMAGE, WorkspacePreviewKind.IMAGE),
                _port("upscaled_condition", "SR/upscaled condition",
                      WorkspaceArtifactKind.IMAGE, WorkspacePreviewKind.IMAGE),
            ),
            "local_condition", "Prepare and approve the image used by local generation.",
        ),
        PipelineOperationSpec(
            "local.generate_geometry", "Generate isolated local geometry", "local",
            ("upscaled_condition", "roi"),
            (
                _port("local_checkpoint", "Local shape checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("local_sparse_preview", "Local sparse preview",
                      WorkspaceArtifactKind.VOLUME, WorkspacePreviewKind.VOLUME),
                _port("local_lr_preview", "Local LR preview",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
                _port("local_mesh", "Raw local 1536 mesh", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
            ),
            "local_geometry", "Generate the selected region in the complete local grid.",
        ),
        PipelineOperationSpec(
            "local.register", "Register local geometry", "local",
            ("local_mesh", "roi", "hr_mesh"),
            (
                _port("registered_local_mesh", "Registered local mesh",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
                _port("registration_overlay", "Base/local registration overlay",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.OVERLAY),
                _port("registration", "Registration transform",
                      WorkspaceArtifactKind.METADATA),
            ),
            "local_registration", "Inspect scale and placement before fusion.",
        ),
        PipelineOperationSpec(
            "local.fuse", "Fuse local and base geometry", "local",
            ("hr_mesh", "registered_local_mesh", "roi"),
            (
                _port("overlap_preview", "Overlap collar diagnostic",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.OVERLAY),
                _port("fused_mesh", "Raw fused mesh", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
            ),
            "geometry", "Join the accepted local detail to the base mesh.",
        ),
        PipelineOperationSpec(
            "geometry.remesh", "Remesh / decimate", "geometry", ("fused_mesh",),
            (_port("processed_mesh", "Processed mesh", WorkspaceArtifactKind.MESH,
                   WorkspacePreviewKind.MESH),),
            "geometry", "Produce a controllable topology variant.",
        ),
        PipelineOperationSpec(
            "geometry.normals", "Generate smooth normals", "geometry",
            ("processed_mesh",),
            (_port("normal_mesh", "Mesh with normals", WorkspaceArtifactKind.MESH,
                   WorkspacePreviewKind.MESH),),
            "geometry", "Generate or preserve shading normals independently of remesh.",
        ),
        PipelineOperationSpec(
            "texture.generate", "Generate base texture", "texture",
            ("hr_checkpoint", "normal_mesh", "source"),
            (
                _port("texture_checkpoint", "Texture checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("texture_atlas", "Texture atlas", WorkspaceArtifactKind.TEXTURE,
                      WorkspacePreviewKind.IMAGE),
                _port("textured_mesh", "Textured mesh", WorkspaceArtifactKind.MESH,
                      WorkspacePreviewKind.MESH),
            ),
            "texture", "Generate a texture package for accepted geometry.",
        ),
        PipelineOperationSpec(
            "texture.refine", "Refine texture", "texture",
            ("texture_checkpoint", "refine_mask", "source"),
            (
                _port("texture_checkpoint", "Refined texture checkpoint",
                      WorkspaceArtifactKind.CHECKPOINT),
                _port("texture_atlas", "Refined texture atlas",
                      WorkspaceArtifactKind.TEXTURE, WorkspacePreviewKind.IMAGE),
                _port("textured_mesh", "Refined textured mesh",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
            ),
            "texture", "Create a masked or global texture variant.",
        ),
        PipelineOperationSpec(
            "texture.local_generate", "Texture isolated local geometry", "texture",
            ("local_checkpoint", "upscaled_condition", "registered_local_mesh"),
            (
                _port("local_texture_maps", "Local texture maps",
                      WorkspaceArtifactKind.TEXTURE, WorkspacePreviewKind.IMAGE),
                _port("textured_local_mesh", "Textured local mesh",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
            ),
            "local_texture", "Inspect local texturing before transfer.",
        ),
        PipelineOperationSpec(
            "texture.transfer", "Transfer and bake local texture", "texture",
            ("textured_mesh", "textured_local_mesh", "normal_mesh", "roi"),
            (
                _port("blend_weights", "Blend-weight map",
                      WorkspaceArtifactKind.TEXTURE, WorkspacePreviewKind.IMAGE),
                _port("transferred_atlas", "Transferred texture atlas",
                      WorkspaceArtifactKind.TEXTURE, WorkspacePreviewKind.IMAGE),
                _port("transferred_mesh", "Transferred textured mesh",
                      WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
            ),
            "texture", "Transfer accepted local appearance onto fused geometry.",
        ),
        PipelineOperationSpec(
            "final.assemble", "Assemble final asset", "final",
            ("normal_mesh", "texture_atlas"),
            (_port("final_mesh", "Final textured asset", WorkspaceArtifactKind.MESH,
                   WorkspacePreviewKind.MESH),),
            "final", "Assemble explicitly accepted geometry and texture variants.",
        ),
    ),
)


RECONSTRUCTION_PIPELINES = {
    PIXAL3D_PIPELINE.backend: PIXAL3D_PIPELINE,
}


LEGACY_OPERATION_TARGET_STAGES = {
    "source.prepare": ReconstructionStage.SOURCE_IMAGE,
    "sparse.generate": ReconstructionStage.SPARSE_OCCUPANCY,
    "lr.generate": ReconstructionStage.LR_SHAPE_LATENT,
    "hr.coordinates": ReconstructionStage.HR_COORDINATES,
    "hr.generate": ReconstructionStage.HR_SHAPE_LATENT,
    "texture.generate": ReconstructionStage.TEXTURE_LATENT,
    "final.assemble": ReconstructionStage.FINAL_MESH,
}

# Parameters exposed by the experimental inspector for each currently
# executable Pixal3D operation. Diffusion phases own independent seed/steps;
# deterministic transform/decode/assembly operations intentionally do not.
PIXAL3D_OPERATION_PARAMETER_KEYS = {
    "source.prepare": ("manual_fov_degrees",),
    "sparse.generate": ("pixal3d_sparse_seed", "pixal3d_sparse_steps"),
    "lr.generate": (
        "pixal3d_lr_seed", "pixal3d_lr_steps",
        "lr_conditioning_resolution",
    ),
    "hr.coordinates": ("resolution",),
    "hr.generate": ("pixal3d_hr_seed", "pixal3d_hr_steps"),
    "texture.generate": (
        "pixal3d_texture_seed", "pixal3d_texture_steps",
    ),
    "final.assemble": ("decimation_target", "texture_size"),
}


def pixal3d_operation_parameters(
    parameters: ReconstructionParameters, operation_key: str
) -> dict[str, object]:
    """Return only parameters which can affect one Pixal3D operation."""
    result = {}
    for key in PIXAL3D_OPERATION_PARAMETER_KEYS.get(operation_key, ()):
        if key.startswith("pixal3d_") and key.endswith("_seed"):
            phase = key.removeprefix("pixal3d_").removesuffix("_seed")
            result[key] = parameters.pixal3d_seed_for(phase)
        elif key.startswith("pixal3d_") and key.endswith("_steps"):
            phase = key.removeprefix("pixal3d_").removesuffix("_steps")
            result[key] = parameters.pixal3d_steps_for(phase)
        else:
            result[key] = getattr(parameters, key)
    return result


_LEGACY_BASE_OPERATIONS = (
    ("source.prepare", (ReconstructionStage.SOURCE_IMAGE,)),
    ("sparse.generate", (ReconstructionStage.SPARSE_OCCUPANCY,)),
    (
        "lr.generate",
        (ReconstructionStage.LR_SHAPE_FLOW,
         ReconstructionStage.LR_SHAPE_LATENT),
    ),
    ("hr.coordinates", (ReconstructionStage.HR_COORDINATES,)),
    (
        "hr.generate",
        (ReconstructionStage.HR_SHAPE_FLOW,
         ReconstructionStage.HR_SHAPE_LATENT),
    ),
    (
        "texture.generate",
        (ReconstructionStage.TEXTURE_FLOW,
         ReconstructionStage.TEXTURE_LATENT),
    ),
    ("final.assemble", (ReconstructionStage.FINAL_MESH,)),
)

_LEGACY_REFINE_OPERATIONS = (
    (
        "hr.refine",
        (ReconstructionStage.HR_SHAPE_FLOW,
         ReconstructionStage.HR_SHAPE_LATENT),
    ),
    (
        "texture.generate",
        (ReconstructionStage.TEXTURE_FLOW,
         ReconstructionStage.TEXTURE_LATENT),
    ),
    ("final.assemble", (ReconstructionStage.FINAL_MESH,)),
)

_LEGACY_TEXTURE_REFINE_OPERATIONS = (
    (
        "texture.refine",
        (ReconstructionStage.TEXTURE_FLOW,
         ReconstructionStage.TEXTURE_LATENT),
    ),
    ("final.assemble", (ReconstructionStage.FINAL_MESH,)),
)

_LEGACY_STAGE_OUTPUT_ROLES = {
    ReconstructionStage.SOURCE_IMAGE: "source",
    ReconstructionStage.SPARSE_OCCUPANCY: "sparse_preview",
    ReconstructionStage.LR_SHAPE_LATENT: "lr_preview",
    ReconstructionStage.HR_COORDINATES: "hr_coordinate_preview",
    ReconstructionStage.HR_SHAPE_LATENT: "hr_mesh",
    ReconstructionStage.TEXTURE_LATENT: "textured_mesh",
    ReconstructionStage.FINAL_MESH: "final_mesh",
}


def _legacy_status(
        statuses: Mapping[ReconstructionStage, ReconstructionStageStatus],
        stages: Sequence[ReconstructionStage],
) -> WorkspaceOperationStatus:
    values = tuple(statuses.get(stage, ReconstructionStageStatus.PENDING)
                   for stage in stages)
    if ReconstructionStageStatus.FAILED in values:
        return WorkspaceOperationStatus.FAILED
    if ReconstructionStageStatus.RUNNING in values:
        return WorkspaceOperationStatus.RUNNING
    if any(value is ReconstructionStageStatus.READY for value in values):
        return WorkspaceOperationStatus.READY
    return WorkspaceOperationStatus.PLANNED


def _legacy_artifact_kind(
        artifact: ReconstructionStageArtifact,
) -> tuple[WorkspaceArtifactKind, WorkspacePreviewKind]:
    return {
        "image": (WorkspaceArtifactKind.IMAGE, WorkspacePreviewKind.IMAGE),
        "points": (
            WorkspaceArtifactKind.POINT_CLOUD,
            WorkspacePreviewKind.POINTS,
        ),
        "mesh": (WorkspaceArtifactKind.MESH, WorkspacePreviewKind.MESH),
        "latent": (WorkspaceArtifactKind.LATENT, WorkspacePreviewKind.NONE),
    }[artifact.preview_kind]


def _legacy_run_label(run: ReconstructionRun, index: int) -> str:
    if run.kind is ReconstructionRunKind.BASE:
        return "Base"
    if run.kind is ReconstructionRunKind.MASKED_TEXTURE_REFINE:
        return f"Texture refine {index}"
    return f"Geometry refine {index}"


def build_legacy_workspace(
        parameters: ReconstructionParameters,
        stage_statuses: Mapping[
            ReconstructionStage, ReconstructionStageStatus
        ],
        stage_artifacts: Mapping[
            ReconstructionStage, ReconstructionStageArtifact
        ],
        runs: Sequence[ReconstructionRun],
        active_run_id: str | None,
        *,
        lr_variants: Sequence[ReconstructionLrVariant] = (),
        selected_lr_refine_source_id: str | None = None,
) -> ReconstructionWorkspace | None:
    """Project the working legacy Pixal3D state into an isolated graph.

    The projection intentionally has no write-back path. Paths without a real
    content hash receive deterministic legacy IDs but cannot be reused as a
    verified content-addressed cache entry.
    """
    if parameters.backend is not ReconstructionBackend.PIXAL3D:
        return None
    workspace = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    if runs:
        snapshots = []
        refine_index = 0
        for run in runs:
            if run.kind is not ReconstructionRunKind.BASE:
                refine_index += 1
            statuses = (
                dict(zip(RECONSTRUCTION_STAGES, run.stage_statuses))
                if len(run.stage_statuses) == len(RECONSTRUCTION_STAGES)
                else {
                    stage: ReconstructionStageStatus.PENDING
                    for stage in RECONSTRUCTION_STAGES
                }
            )
            artifacts = {item.stage: item for item in run.stage_artifacts}
            artifacts.setdefault(
                ReconstructionStage.FINAL_MESH,
                ReconstructionStageArtifact(
                    ReconstructionStage.FINAL_MESH, run.glb_path, "mesh"
                ),
            )
            if run.source_path:
                artifacts.setdefault(
                    ReconstructionStage.SOURCE_IMAGE,
                    ReconstructionStageArtifact(
                        ReconstructionStage.SOURCE_IMAGE,
                        run.source_path,
                        "image",
                    ),
                )
            snapshots.append((
                run.run_id,
                _legacy_run_label(run, refine_index),
                run.kind,
                statuses,
                artifacts,
                run,
                None,
            ))
    else:
        base_lr_variant = next(
            (item for item in lr_variants if item.parent_variant_id is None),
            None,
        )
        current_artifacts = dict(stage_artifacts)
        if base_lr_variant is not None:
            current_artifacts[ReconstructionStage.LR_SHAPE_LATENT] = (
                base_lr_variant.preview_artifact
            )
        snapshots = [(
            "current",
            "Current generation",
            ReconstructionRunKind.BASE,
            dict(stage_statuses),
            current_artifacts,
            None,
            base_lr_variant,
        )]

    for run_id, label, kind, statuses, artifacts, run, lr_variant in snapshots:
        operation_specs = {
            ReconstructionRunKind.BASE: _LEGACY_BASE_OPERATIONS,
            ReconstructionRunKind.MASKED_REFINE: _LEGACY_REFINE_OPERATIONS,
            ReconstructionRunKind.MASKED_TEXTURE_REFINE: (
                _LEGACY_TEXTURE_REFINE_OPERATIONS
            ),
        }[kind]
        preceding_outputs: list[str] = []
        for spec_key, stages in operation_specs:
            operation = workspace.plan_operation(
                spec_key,
                input_artifact_ids=tuple(preceding_outputs[-4:]),
                parameters=(
                    pixal3d_operation_parameters(parameters, spec_key)
                    if spec_key in PIXAL3D_OPERATION_PARAMETER_KEYS
                    else parameters.to_dict()
                ),
                model_identity="legacy:pixal3d",
                worker_protocol="legacy-stage-events-v1",
                variant_label=label,
                operation_id=f"legacy:{run_id}:{spec_key}",
            )
            published_any = False
            for stage in stages:
                artifact = artifacts.get(stage)
                role = _LEGACY_STAGE_OUTPUT_ROLES.get(stage)
                if artifact is None or role is None:
                    continue
                artifact_kind, preview_kind = _legacy_artifact_kind(artifact)
                published = workspace.publish_artifact(
                    operation.operation_id,
                    role,
                    path=artifact.path,
                    kind=artifact_kind,
                    preview_kind=preview_kind,
                    artifact_id=(
                        f"legacy:{run_id}:{spec_key}:{stage.value}"
                    ),
                    metadata={
                        "legacy_stage": stage.value,
                        "verified_content_hash": False,
                    },
                )
                preceding_outputs.append(published.artifact_id)
                published_any = True
            if run is not None and spec_key in {"hr.generate", "hr.refine"}:
                if run.checkpoint_path:
                    checkpoint = workspace.publish_artifact(
                        operation.operation_id,
                        "hr_checkpoint",
                        path=run.checkpoint_path,
                        artifact_id=(
                            f"legacy:{run_id}:{spec_key}:hr_checkpoint"
                        ),
                        metadata={"verified_content_hash": False},
                    )
                    preceding_outputs.append(checkpoint.artifact_id)
                    published_any = True
            if lr_variant is not None and spec_key == "lr.generate":
                checkpoint = workspace.publish_artifact(
                    operation.operation_id,
                    "lr_checkpoint",
                    path=lr_variant.checkpoint_path,
                    artifact_id=(
                        f"legacy:{run_id}:{spec_key}:lr_checkpoint"
                    ),
                    metadata={
                        "verified_content_hash": False,
                        "lr_variant_id": lr_variant.variant_id,
                        "lr_variant_label": lr_variant.label,
                        "selected_refine_source": (
                            lr_variant.variant_id
                            == selected_lr_refine_source_id
                        ),
                    },
                )
                preceding_outputs.append(checkpoint.artifact_id)
                published_any = True
            if run is not None and spec_key in {
                    "texture.generate", "texture.refine"}:
                if run.texture_checkpoint_path:
                    checkpoint = workspace.publish_artifact(
                        operation.operation_id,
                        "texture_checkpoint",
                        path=run.texture_checkpoint_path,
                        artifact_id=(
                            f"legacy:{run_id}:{spec_key}:texture_checkpoint"
                        ),
                        metadata={"verified_content_hash": False},
                    )
                    preceding_outputs.append(checkpoint.artifact_id)
                    published_any = True
            status = _legacy_status(statuses, stages)
            if (
                    status is WorkspaceOperationStatus.PLANNED
                    and published_any):
                status = WorkspaceOperationStatus.READY
            if spec_key == "final.assemble" and run is not None:
                status = WorkspaceOperationStatus.READY
            workspace.set_operation_status(operation.operation_id, status)

    refined_lr_variants = tuple(
        item for item in lr_variants if item.parent_variant_id is not None
    )
    for variant in refined_lr_variants:
        operation = workspace.plan_operation(
            "lr.refine",
            parameters=parameters.to_dict(),
            model_identity="legacy:pixal3d",
            worker_protocol="legacy-stage-events-v1",
            variant_label=variant.label,
            operation_id=f"legacy:{variant.variant_id}:lr.refine",
        )
        preview = workspace.publish_artifact(
            operation.operation_id,
            "lr_preview",
            path=variant.preview_artifact.path,
            kind=WorkspaceArtifactKind.MESH,
            preview_kind=WorkspacePreviewKind.MESH,
            artifact_id=(
                f"legacy:{variant.variant_id}:lr.refine:lr_shape_latent"
            ),
            metadata={
                "legacy_stage": ReconstructionStage.LR_SHAPE_LATENT.value,
                "verified_content_hash": False,
                "lr_variant_id": variant.variant_id,
            },
        )
        workspace.publish_artifact(
            operation.operation_id,
            "lr_checkpoint",
            path=variant.checkpoint_path,
            artifact_id=f"legacy:{variant.variant_id}:lr.refine:lr_checkpoint",
            metadata={
                "verified_content_hash": False,
                "lr_variant_id": variant.variant_id,
                "lr_variant_label": variant.label,
                "selected_refine_source": (
                    variant.variant_id == selected_lr_refine_source_id
                ),
            },
        )
        workspace.set_operation_status(
            operation.operation_id, WorkspaceOperationStatus.READY
        )
        if variant.variant_id == selected_lr_refine_source_id:
            workspace.select_artifact(preview.artifact_id)

    active_prefix = f"legacy:{active_run_id}:" if active_run_id else None
    if active_prefix:
        for operation in workspace.operations:
            if operation.operation_id.startswith(active_prefix):
                artifacts = workspace.artifacts_for_operation(
                    operation.operation_id
                )
                preview = next(
                    (item for item in artifacts
                     if item.preview_kind is not WorkspacePreviewKind.NONE),
                    None,
                )
                if preview is not None:
                    workspace.select_artifact(preview.artifact_id)
    return workspace
