"""Toolkit-neutral state and intents for generation tool panels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import random
from typing import Callable, Protocol

from ..canvas.editor_canvas_controller import EditorCanvasController
from ..document.commands import (
    ClearIpAdapterReferenceLayerCommand,
    ClearLayerMaskCommand,
    SetIpAdapterReferenceLayerCommand,
    UpdateDiffusionToolCommand,
    UpdateInstructToolCommand,
)
from ..document.change_event import DocumentChangeEvent
from ..document.layer import Layer
from ..document.tool import DiffusionTool, InstructTool, LamaTool
from .application import EditorApplication
from .presentation import PanelUpdate


class GenerationPanelKind(str, Enum):
    NONE = "none"
    DIFFUSION = "diffusion"
    LAMA = "lama"
    INSTRUCT = "instruct"


class GenerationPhase(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    RESULT = "result"
    ERROR = "error"


class GenerationAction(str, Enum):
    SELECT_MODEL = "select_model"
    SELECT_PREDICTION = "select_prediction"
    SET_PROMPT = "set_prompt"
    SET_NEGATIVE_PROMPT = "set_negative_prompt"
    SET_MODE = "set_mode"
    SET_MASKED_CONTENT = "set_masked_content"
    SET_STRENGTH = "set_strength"
    SET_STEPS = "set_steps"
    SET_GUIDANCE = "set_guidance"
    SET_RESIZE = "set_resize"
    SET_SEED = "set_seed"
    RANDOM_SEED = "random_seed"
    LOAD_MODEL = "load_model"
    RUN = "run"
    LOAD_IP_ADAPTER = "load_ip_adapter"
    SELECT_IP_REFERENCE = "select_ip_reference"
    SET_IP_SCALE = "set_ip_scale"
    SET_INSTRUCTION = "set_instruction"
    SET_IMAGE_GUIDANCE = "set_image_guidance"
    SET_MASK_SIZE = "set_mask_size"
    SET_MASK_HARDNESS = "set_mask_hardness"
    SET_MASK_FLOW = "set_mask_flow"
    SET_MASK_ERASER = "set_mask_eraser"
    SET_SHOW_MASK = "set_show_mask"
    CLEAR_MASK = "clear_mask"
    SELECT_BACKGROUND = "select_background"


@dataclass(frozen=True)
class GenerationIntent:
    action: GenerationAction
    value: object = None


@dataclass(frozen=True)
class ReferenceChoice:
    stable_id: str
    name: str


@dataclass(frozen=True)
class MaskBrushState:
    size: int = 50
    hardness: float = 0.4
    flow: float = 1.0
    eraser: bool = False
    show: bool = True


@dataclass(frozen=True)
class DiffusionPanelState:
    prompt: str = ""
    negative_prompt: str = ""
    strength: float = 0.3
    steps: int = 20
    guidance_scale: float = 7.0
    seed_text: str = "-1"
    mode: str = "inpaint"
    masked_content: str = "original"
    resize_to_model_resolution: bool = False
    prediction_type: str = ""
    selected_model_path: str = ""
    ip_adapter_scale: float = 0.6
    ip_adapter_layer_id: str | None = None
    phase: GenerationPhase = GenerationPhase.IDLE
    message: str = "No model loaded"
    model_diagnostics: str = ""
    ip_adapter_message: str = "Not loaded"
    layer_info: str = "No diffusion layer selected"


@dataclass(frozen=True)
class LamaPanelState:
    phase: GenerationPhase = GenerationPhase.IDLE
    message: str = "Ready"
    layer_info: str = "No LaMa layer selected"


@dataclass(frozen=True)
class InstructPanelState:
    instruction: str = ""
    image_guidance_scale: float = 1.5
    guidance_scale: float = 7.0
    steps: int = 20
    seed_text: str = "-1"
    phase: GenerationPhase = GenerationPhase.IDLE
    message: str = "Not loaded"
    layer_info: str = "No instruct layer selected"


@dataclass(frozen=True)
class GenerationPanelsState:
    active_kind: GenerationPanelKind
    active_layer_id: str | None
    models: tuple[ReferenceChoice, ...]
    reference_layers: tuple[ReferenceChoice, ...]
    mask: MaskBrushState
    diffusion: DiffusionPanelState
    lama: LamaPanelState
    instruct: InstructPanelState


class GenerationPanelsPresentation(Protocol):
    def apply_generation_panels_state(
            self, state: GenerationPanelsState) -> None: ...


class GenerationPanelsCoordinator:
    """Owns generation panel drafts and bridges explicit intents to controllers."""

    def __init__(
            self,
            application: EditorApplication,
            canvas: EditorCanvasController,
            *,
            canvas_controls=None,
            random_seed: Callable[[], int] | None = None) -> None:
        self._application = application
        self._stack = application.layer_stack
        self._document = application.document
        self._canvas = canvas
        self._canvas_controls = canvas_controls
        self._random_seed = random_seed or (
            lambda: random.randint(0, 2**32 - 1))
        self._view: GenerationPanelsPresentation | None = None
        self._closed = False
        self._mask = MaskBrushState()
        self._models = self._scan_models()
        self._diffusion_drafts: dict[
            str, tuple[int, DiffusionPanelState]] = {}
        self._instruct_drafts: dict[
            str, tuple[int, InstructPanelState]] = {}
        self._diffusion_phase = (
            GenerationPhase.READY
            if bool(getattr(application.engines.diffusion, "is_loaded", False))
            else GenerationPhase.IDLE
        )
        self._diffusion_message = self._initial_diffusion_message()
        self._diffusion_diagnostics = ""
        self._ip_adapter_message = (
            "Loaded"
            if bool(getattr(application.engines.diffusion,
                            "ip_adapter_loaded", False))
            else "Not loaded"
        )
        self._lama_phase = GenerationPhase.IDLE
        self._lama_message = "Ready"
        self._instruct_phase = (
            GenerationPhase.READY
            if bool(getattr(application.engines.instruct, "is_loaded", False))
            else GenerationPhase.IDLE
        )
        self._instruct_message = (
            "Loaded: instruct-pix2pix"
            if self._instruct_phase == GenerationPhase.READY
            else "Not loaded"
        )
        self._stack_subscription = self._stack.subscribe(self._on_stack_changed)
        self._state = self._build_state()

    @property
    def state(self) -> GenerationPanelsState:
        return self._state

    def bind_view(self, view: GenerationPanelsPresentation) -> None:
        self._require_open()
        self._view = view
        view.apply_generation_panels_state(self._state)

    def update_panel(self, update: PanelUpdate) -> None:
        """PanelPresentation port used by EditorApplication.poll."""
        if self._closed:
            return
        message = str(update.payload.get("error", ""))
        if update.panel_id == "diffusion":
            if update.state == "model-loading":
                self._diffusion_phase = GenerationPhase.LOADING
                self._diffusion_message = "Loading..."
            elif update.state == "model-error":
                self._diffusion_phase = GenerationPhase.ERROR
                self._diffusion_message = f"Error: {message[:80]}"
                self._diffusion_diagnostics = ""
            elif update.state == "model-loaded":
                self._diffusion_phase = GenerationPhase.READY
                path = str(update.payload.get("path", ""))
                self._diffusion_message = (
                    f"Loaded: {os.path.basename(path)}" if path else "Loaded")
                info = update.payload.get("info", {})
                if isinstance(info, dict):
                    diagnostics = (
                        f"scheduler: {info.get('scheduler', '?')}\n"
                        f"prediction: {info.get('prediction_type', '?')}\n"
                        f"algorithm: {info.get('algorithm_type', '?')}"
                    )
                    self._set_diffusion_diagnostics(diagnostics)
            elif update.state == "ip-adapter-error":
                self._ip_adapter_message = f"Error: {message[:60]}"
            elif update.state == "ip-adapter-loaded":
                self._ip_adapter_message = "Loaded"
            elif update.state == "running":
                self._diffusion_phase = GenerationPhase.RUNNING
                self._diffusion_message = str(
                    update.payload.get("status", "Running..."))
            elif update.state == "result":
                self._diffusion_phase = GenerationPhase.RESULT
                self._diffusion_message = "Result applied"
            elif update.state == "inference-error":
                self._diffusion_phase = GenerationPhase.ERROR
                self._diffusion_message = f"Error: {message[:80]}"
        elif update.panel_id == "lama":
            if update.state == "running":
                self._lama_phase = GenerationPhase.RUNNING
                self._lama_message = str(
                    update.payload.get("status", "Running..."))
            elif update.state == "result":
                self._lama_phase = GenerationPhase.RESULT
                self._lama_message = "Result applied"
            elif update.state == "error":
                self._lama_phase = GenerationPhase.ERROR
                self._lama_message = f"Error: {message[:80]}"
        elif update.panel_id == "instruct":
            if update.state == "model-loading":
                self._instruct_phase = GenerationPhase.LOADING
                self._instruct_message = "Loading..."
            elif update.state == "model-error":
                self._instruct_phase = GenerationPhase.ERROR
                self._instruct_message = f"Error: {message[:60]}"
            elif update.state == "model-loaded":
                self._instruct_phase = GenerationPhase.READY
                self._instruct_message = "Loaded: instruct-pix2pix"
            elif update.state == "running":
                self._instruct_phase = GenerationPhase.RUNNING
                self._instruct_message = str(
                    update.payload.get("status", "Running..."))
            elif update.state == "result":
                self._instruct_phase = GenerationPhase.RESULT
                self._instruct_message = "Result applied"
            elif update.state == "inference-error":
                self._instruct_phase = GenerationPhase.ERROR
                self._instruct_message = f"Error: {message[:80]}"
        self.refresh()

    def handle_intent(self, intent: GenerationIntent) -> None:
        self._require_open()
        action, value = intent.action, intent.value
        layer = self._stack.active_layer
        if action in {
                GenerationAction.SET_PROMPT,
                GenerationAction.SET_NEGATIVE_PROMPT,
                GenerationAction.SET_MODE,
                GenerationAction.SET_MASKED_CONTENT,
                GenerationAction.SET_STRENGTH,
                GenerationAction.SET_STEPS,
                GenerationAction.SET_GUIDANCE,
                GenerationAction.SET_RESIZE,
                GenerationAction.SET_SEED,
                GenerationAction.SELECT_PREDICTION,
                GenerationAction.SELECT_MODEL,
                GenerationAction.SET_IP_SCALE}:
            if isinstance(getattr(layer, "tool", None), DiffusionTool):
                self._edit_diffusion(layer, action, value)
            elif (
                    action in {
                        GenerationAction.SET_STEPS,
                        GenerationAction.SET_GUIDANCE,
                        GenerationAction.SET_SEED,
                    }
                    and isinstance(
                        getattr(layer, "tool", None), InstructTool)):
                self._edit_instruct(layer, action, value)
        elif action in {
                GenerationAction.SET_INSTRUCTION,
                GenerationAction.SET_IMAGE_GUIDANCE}:
            self._edit_instruct(layer, action, value)
        elif action == GenerationAction.RANDOM_SEED:
            self._set_random_seed(layer)
        elif action == GenerationAction.LOAD_MODEL:
            self._load_model(layer)
        elif action == GenerationAction.RUN:
            self._run(layer)
        elif action == GenerationAction.LOAD_IP_ADAPTER:
            self._load_ip_adapter()
        elif action == GenerationAction.SELECT_IP_REFERENCE:
            self._select_ip_reference(layer, value)
        elif action == GenerationAction.SET_MASK_SIZE:
            self._mask = replace(
                self._mask, size=max(1, min(int(value), 500)))
            self._apply_mask_brush()
        elif action == GenerationAction.SET_MASK_HARDNESS:
            self._mask = replace(
                self._mask,
                hardness=max(0.0, min(float(value), 1.0)))
            self._apply_mask_brush()
        elif action == GenerationAction.SET_MASK_FLOW:
            self._mask = replace(
                self._mask, flow=max(0.0, min(float(value), 1.0)))
            self._apply_mask_brush()
        elif action == GenerationAction.SET_MASK_ERASER:
            self._mask = replace(self._mask, eraser=bool(value))
            self._apply_mask_brush()
        elif action == GenerationAction.SET_SHOW_MASK:
            self._mask = replace(self._mask, show=bool(value))
            self._canvas.set_show_mask(self._mask.show)
        elif action == GenerationAction.CLEAR_MASK:
            if layer is not None and isinstance(
                    layer.tool, (DiffusionTool, LamaTool, InstructTool)):
                self._document.execute(ClearLayerMaskCommand(
                    layer=layer,
                    label=f"Clear {layer.tool.tool_type.title()} Mask",
                ))
        elif action == GenerationAction.SELECT_BACKGROUND:
            if layer is not None and isinstance(
                    layer.tool, (DiffusionTool, LamaTool)):
                event = (
                    self._application.segmentation_controller
                    .start_select_background(layer)
                )
                self._handle_immediate_status(event.status)
        else:
            raise ValueError(f"unsupported generation action: {action}")
        self.refresh()

    def refresh(self) -> None:
        if self._closed:
            return
        self._state = self._build_state()
        if self._view is not None:
            self._view.apply_generation_panels_state(self._state)

    def refresh_models(self) -> None:
        if self._closed:
            return
        self._models = self._scan_models()
        self.refresh()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._view = None
        self._diffusion_drafts.clear()
        self._instruct_drafts.clear()
        self._stack_subscription.unsubscribe()

    def _build_state(self) -> GenerationPanelsState:
        layer = self._stack.active_layer
        tool = layer.tool if layer is not None else None
        live_ids = {
            candidate.id for candidate in self._stack.all_layers()}
        self._diffusion_drafts = {
            stable_id: draft
            for stable_id, draft in self._diffusion_drafts.items()
            if stable_id in live_ids
        }
        self._instruct_drafts = {
            stable_id: draft
            for stable_id, draft in self._instruct_drafts.items()
            if stable_id in live_ids
        }
        kind = GenerationPanelKind.NONE
        if isinstance(tool, DiffusionTool):
            kind = GenerationPanelKind.DIFFUSION
        elif isinstance(tool, LamaTool):
            kind = GenerationPanelKind.LAMA
        elif isinstance(tool, InstructTool):
            kind = GenerationPanelKind.INSTRUCT
        models = tuple(
            ReferenceChoice(path, os.path.basename(path))
            for path in self._models
        )
        references = tuple(
            ReferenceChoice(candidate.id, candidate.name)
            for candidate in self._stack.all_layers()
            if layer is None or candidate.id != layer.id
        )
        diffusion = self._diffusion_state(layer, tool)
        instruct = self._instruct_state(layer, tool)
        lama = LamaPanelState(
            phase=self._lama_phase,
            message=self._lama_message,
            layer_info=self._layer_info(layer, tool)
            if isinstance(tool, LamaTool)
            else "No LaMa layer selected",
        )
        return GenerationPanelsState(
            active_kind=kind,
            active_layer_id=layer.id if layer is not None else None,
            models=models,
            reference_layers=references,
            mask=self._mask,
            diffusion=diffusion,
            lama=lama,
            instruct=instruct,
        )

    def _diffusion_state(
            self, layer: Layer | None, tool) -> DiffusionPanelState:
        if layer is None or not isinstance(tool, DiffusionTool):
            return DiffusionPanelState(
                phase=self._diffusion_phase,
                message=self._diffusion_message,
                ip_adapter_message=self._ip_adapter_message,
            )
        cached = self._diffusion_drafts.get(layer.id)
        if cached is None or cached[0] != id(tool):
            draft = DiffusionPanelState(
                prompt=tool.prompt,
                negative_prompt=tool.negative_prompt,
                strength=tool.strength,
                steps=tool.steps,
                guidance_scale=tool.guidance_scale,
                seed_text=str(tool.seed),
                mode=tool.mode,
                masked_content=tool.masked_content,
                resize_to_model_resolution=tool.resize_to_model_resolution,
                prediction_type=tool.prediction_type,
                selected_model_path=(
                    tool.model_path
                    or str(getattr(
                        self._application.engines.diffusion,
                        "model_path", "") or "")
                    or (self._models[0] if self._models else "")
                ),
                ip_adapter_scale=tool.ip_adapter_scale,
                ip_adapter_layer_id=tool.ip_adapter_layer_id,
            )
        else:
            draft = cached[1]
        draft = replace(
            draft,
            phase=self._diffusion_phase,
            message=self._diffusion_message,
            model_diagnostics=getattr(
                self, "_diffusion_diagnostics", ""),
            ip_adapter_message=self._ip_adapter_message,
            layer_info=self._layer_info(layer, tool),
            ip_adapter_layer_id=tool.ip_adapter_layer_id,
        )
        self._diffusion_drafts[layer.id] = (id(tool), draft)
        return draft

    def _instruct_state(
            self, layer: Layer | None, tool) -> InstructPanelState:
        if layer is None or not isinstance(tool, InstructTool):
            return InstructPanelState(
                phase=self._instruct_phase,
                message=self._instruct_message,
            )
        cached = self._instruct_drafts.get(layer.id)
        if cached is None or cached[0] != id(tool):
            draft = InstructPanelState(
                instruction=tool.instruction,
                image_guidance_scale=tool.image_guidance_scale,
                guidance_scale=tool.guidance_scale,
                steps=tool.steps,
                seed_text=str(tool.seed),
            )
        else:
            draft = cached[1]
        draft = replace(
            draft,
            phase=self._instruct_phase,
            message=self._instruct_message,
            layer_info=self._layer_info(layer, tool),
        )
        self._instruct_drafts[layer.id] = (id(tool), draft)
        return draft

    def _edit_diffusion(
            self, layer: Layer | None, action: GenerationAction, value) -> None:
        if layer is None or not isinstance(layer.tool, DiffusionTool):
            return
        state = self._diffusion_state(layer, layer.tool)
        if action == GenerationAction.SET_PROMPT:
            field, normalized = "prompt", str(value)
        elif action == GenerationAction.SET_NEGATIVE_PROMPT:
            field, normalized = "negative_prompt", str(value)
        elif action == GenerationAction.SET_MODE:
            field, normalized = "mode", str(value)
        elif action == GenerationAction.SET_MASKED_CONTENT:
            field, normalized = "masked_content", str(value)
        elif action == GenerationAction.SET_STRENGTH:
            field, normalized = (
                "strength", max(0.0, min(float(value), 1.0)))
        elif action == GenerationAction.SET_STEPS:
            field, normalized = "steps", max(1, min(int(value), 50))
        elif action == GenerationAction.SET_GUIDANCE:
            field, normalized = (
                "guidance_scale",
                max(1.0, min(float(value), 20.0)),
            )
        elif action == GenerationAction.SET_RESIZE:
            field, normalized = "resize_to_model_resolution", bool(value)
        elif action == GenerationAction.SET_SEED:
            field, normalized = "seed_text", str(value)
        elif action == GenerationAction.SELECT_PREDICTION:
            field, normalized = "prediction_type", str(value)
        elif action == GenerationAction.SELECT_MODEL:
            field, normalized = "selected_model_path", str(value)
        elif action == GenerationAction.SET_IP_SCALE:
            field, normalized = (
                "ip_adapter_scale",
                max(0.0, min(float(value), 1.0)),
            )
        else:
            return
        self._diffusion_drafts[layer.id] = (
            id(layer.tool), replace(state, **{field: normalized}))

    def _edit_instruct(
            self, layer: Layer | None, action: GenerationAction, value) -> None:
        if layer is None or not isinstance(layer.tool, InstructTool):
            return
        state = self._instruct_state(layer, layer.tool)
        if action == GenerationAction.SET_INSTRUCTION:
            state = replace(state, instruction=str(value))
        elif action == GenerationAction.SET_IMAGE_GUIDANCE:
            state = replace(
                state,
                image_guidance_scale=max(
                    1.0, min(float(value), 3.0)),
            )
        elif action == GenerationAction.SET_GUIDANCE:
            state = replace(
                state,
                guidance_scale=max(1.0, min(float(value), 20.0)),
            )
        elif action == GenerationAction.SET_STEPS:
            state = replace(
                state, steps=max(1, min(int(value), 50)))
        elif action == GenerationAction.SET_SEED:
            state = replace(state, seed_text=str(value))
        self._instruct_drafts[layer.id] = (id(layer.tool), state)

    def _set_random_seed(self, layer: Layer | None) -> None:
        if layer is None:
            return
        seed = str(self._random_seed())
        if isinstance(layer.tool, DiffusionTool):
            self._edit_diffusion(
                layer, GenerationAction.SET_SEED, seed)
        elif isinstance(layer.tool, InstructTool):
            state = self._instruct_state(layer, layer.tool)
            self._instruct_drafts[layer.id] = (
                id(layer.tool), replace(state, seed_text=seed))

    def _load_model(self, layer: Layer | None) -> None:
        if isinstance(getattr(layer, "tool", None), DiffusionTool):
            draft = self._diffusion_state(layer, layer.tool)
            if not draft.selected_model_path:
                return
            event = self._application.diffusion_controller.submit_load_model(
                draft.selected_model_path,
                draft.prediction_type,
            )
            if event.status:
                self._diffusion_phase = GenerationPhase.LOADING
                self._diffusion_message = event.status
                self._handle_immediate_status(event.status)
        elif isinstance(getattr(layer, "tool", None), InstructTool):
            event = (
                self._application.instruct_controller.submit_load_model())
            if event.model_loading:
                self._instruct_phase = GenerationPhase.LOADING
                self._instruct_message = event.status or "Loading..."
            self._handle_immediate_status(event.status)

    def _run(self, layer: Layer | None) -> None:
        if layer is None:
            return
        if isinstance(layer.tool, DiffusionTool):
            draft = self._diffusion_state(layer, layer.tool)
            seed = self._parse_seed(draft.seed_text)
            if seed < 0:
                seed = self._random_seed()
                self._edit_diffusion(
                    layer, GenerationAction.SET_SEED, str(seed))
                draft = self._diffusion_state(layer, layer.tool)
            model_path = (
                draft.selected_model_path
                or str(getattr(
                    self._application.engines.diffusion,
                    "model_path", "") or "")
            )
            self._document.execute(UpdateDiffusionToolCommand(
                layer=layer,
                prompt=draft.prompt,
                negative_prompt=draft.negative_prompt,
                strength=draft.strength,
                guidance_scale=draft.guidance_scale,
                steps=draft.steps,
                seed=seed,
                mode=draft.mode,
                masked_content=draft.masked_content,
                ip_adapter_scale=draft.ip_adapter_scale,
                resize_to_model_resolution=draft.resize_to_model_resolution,
                model_path=model_path,
                prediction_type=draft.prediction_type,
            ))
            current = self._stack.find_layer_by_id(layer.id)
            if current is None:
                return
            event = (
                self._application.diffusion_controller
                .start_regeneration(current)
            )
            if event.status:
                self._diffusion_phase = GenerationPhase.RUNNING
                self._diffusion_message = event.status
                self._handle_immediate_status(event.status)
        elif isinstance(layer.tool, LamaTool):
            event = self._application.lama_controller.start_remove(layer)
            if event.status:
                self._lama_phase = GenerationPhase.RUNNING
                self._lama_message = event.status
                self._handle_immediate_status(event.status)
        elif isinstance(layer.tool, InstructTool):
            draft = self._instruct_state(layer, layer.tool)
            seed = self._parse_seed(draft.seed_text)
            if seed < 0:
                seed = self._random_seed()
                state = replace(draft, seed_text=str(seed))
                self._instruct_drafts[layer.id] = (id(layer.tool), state)
            self._document.execute(UpdateInstructToolCommand(
                layer=layer,
                instruction=draft.instruction,
                image_guidance_scale=draft.image_guidance_scale,
                guidance_scale=draft.guidance_scale,
                steps=draft.steps,
                seed=seed,
            ))
            current = self._stack.find_layer_by_id(layer.id)
            if current is None:
                return
            event = self._application.instruct_controller.start_apply(current)
            if event.model_loading:
                self._instruct_phase = GenerationPhase.LOADING
            elif event.status:
                self._instruct_phase = GenerationPhase.RUNNING
            if event.status:
                self._instruct_message = event.status
                self._handle_immediate_status(event.status)

    def _load_ip_adapter(self) -> None:
        event = (
            self._application.diffusion_controller
            .submit_load_ip_adapter()
        )
        if event.status:
            self._ip_adapter_message = event.status
            self._handle_immediate_status(event.status)

    def _select_ip_reference(self, layer: Layer | None, value) -> None:
        if layer is None or not isinstance(layer.tool, DiffusionTool):
            return
        reference_id = str(value) if value else ""
        if not reference_id:
            self._document.execute(
                ClearIpAdapterReferenceLayerCommand(layer=layer))
            return
        reference = self._stack.find_layer_by_id(reference_id)
        if reference is None or reference is layer:
            return
        self._document.execute(SetIpAdapterReferenceLayerCommand(
            layer=layer,
            reference_layer_id=reference.id,
            reference_layer_name_hint=reference.name,
        ))
        self._application.set_status(
            f"IP-Adapter reference: {reference.name}")

    def _apply_mask_brush(self) -> None:
        if self._canvas_controls is not None:
            self._canvas_controls.apply_generation_mask_brush(
                self._mask.size,
                self._mask.hardness,
                self._mask.flow,
                self._mask.eraser,
            )
            return
        self._canvas.set_mask_brush(
            self._mask.size,
            self._mask.hardness,
            self._mask.flow,
        )
        self._canvas.set_mask_eraser(self._mask.eraser)

    def _handle_immediate_status(self, status: str | None) -> None:
        if status:
            self._application.set_status(status)

    def _initial_diffusion_message(self) -> str:
        path = str(getattr(
            self._application.engines.diffusion, "model_path", "") or "")
        return f"Loaded: {os.path.basename(path)}" if path else "No model loaded"

    def _set_diffusion_diagnostics(self, value: str) -> None:
        self._diffusion_diagnostics = value

    def _scan_models(self) -> tuple[str, ...]:
        directory = self._application.models_dir
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return ()
        return tuple(
            os.path.join(directory, name)
            for name in names
            if name.endswith(".safetensors") and "flux" not in name.lower()
        )

    @staticmethod
    def _parse_seed(value: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _layer_info(layer: Layer | None, tool) -> str:
        if layer is None:
            return "No generation layer selected"
        mask_status = "has mask" if layer.has_mask() else "no mask"
        if layer.patch_rect:
            x0, y0, x1, y1 = layer.patch_rect
            patch_info = f"manual {x1 - x0}x{y1 - y0}"
        else:
            patch_info = (
                f"auto {getattr(tool, 'patch_w', 0)}x"
                f"{getattr(tool, 'patch_h', 0)}"
            )
        return f"patch: {patch_info}  mask: {mask_status}"

    def _on_stack_changed(self, _event: DocumentChangeEvent) -> None:
        self.refresh()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("generation panels coordinator is closed")
