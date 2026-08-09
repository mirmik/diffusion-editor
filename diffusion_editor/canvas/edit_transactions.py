"""Undoable transactions for direct canvas edits.

Brush tools deliberately update the document while the pointer is moving so
the canvas can provide immediate feedback.  This coordinator turns that
external mutation into one history entry when the gesture finishes, and can
restore the pre-gesture state when input is cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Protocol

import numpy as np

from ..document.document_service import DocumentService
from ..document.change_event import DocumentChangeKind
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.mask import coerce_mask_data

Rect = tuple[int, int, int, int]
logger = logging.getLogger(__name__)


class CanvasEditSource(Protocol):
    on_edit_begin: Callable[[str, Layer | None, str], None] | None
    on_edit_end: Callable[[Layer | None, str, Rect | None], None] | None
    on_edit_cancel: Callable[[Layer | None, str], None] | None


@dataclass
class _EditContext:
    label: str
    target: str
    layer_id: str | None
    layer: Layer | None
    before_array: np.ndarray | None = None
    before_image: np.ndarray | None = None
    before_offset: tuple[int, int] | None = None


class CanvasEditTransactionCoordinator:
    """Record direct canvas mutations as exception-safe undo callbacks."""

    def __init__(
            self,
            layer_stack: LayerStack,
            document: DocumentService,
            *,
            history_replaying: Callable[[], bool] | None = None,
            on_history_changed: Callable[[], None] | None = None,
            on_edit_cancelled: Callable[[], None] | None = None,
            on_mutation_begin: Callable[[], object] | None = None,
            cancel_interaction: Callable[[], None] | None = None) -> None:
        self._layer_stack = layer_stack
        self._document = document
        self._history_replaying = history_replaying or (lambda: False)
        self._on_history_changed = on_history_changed or (lambda: None)
        self._on_edit_cancelled = on_edit_cancelled or (lambda: None)
        self._on_mutation_begin = on_mutation_begin or (lambda: None)
        self._cancel_interaction = cancel_interaction
        self._context: _EditContext | None = None
        self._source: CanvasEditSource | None = None
        self._previous_callbacks: tuple[object, object, object] | None = None
        self._remove_mutation_listener: Callable[[], None] | None = None
        self._closed = False

    @property
    def active(self) -> bool:
        return self._context is not None

    def bind(self, source: CanvasEditSource) -> None:
        if self._closed:
            raise RuntimeError("canvas edit transaction coordinator is closed")
        if self._source is source:
            return
        if self._source is not None:
            raise RuntimeError("canvas edit transaction coordinator is already bound")
        self._source = source
        self._previous_callbacks = (
            source.on_edit_begin,
            source.on_edit_end,
            source.on_edit_cancel,
        )
        source.on_edit_begin = self.begin
        source.on_edit_end = self.finish
        source.on_edit_cancel = self.cancel
        self._remove_mutation_listener = (
            self._document.add_before_mutation_listener(
                self._cancel_source_interaction
            )
        )

    def begin(
            self,
            label: str,
            layer: Layer | None,
            target: str) -> None:
        if self._closed or self._history_replaying():
            return
        if self._context is not None:
            self._restore(self._context)
            self._context = None

        if target == "selection":
            self._activate_context(_EditContext(
                label=label,
                target=target,
                layer_id=None,
                layer=None,
                before_array=self._layer_stack.selection.data.copy(),
            ))
            return

        if (
                layer is None
                or self._layer_stack.find_layer_by_id(layer.id) is not layer):
            return
        if target == "transform":
            self._activate_context(_EditContext(
                label=label,
                target=target,
                layer_id=layer.id,
                layer=layer,
                before_offset=(layer.x, layer.y),
            ))
            return
        before_image = None
        if target == "mask":
            before = layer.mask.data.copy()
        elif target == "mask_image":
            before = layer.mask.data.copy()
            before_image = layer.image.copy()
        elif target == "image":
            before = layer.image.copy()
        else:
            raise ValueError(f"unsupported canvas edit target: {target}")
        self._activate_context(_EditContext(
            label=label,
            target=target,
            layer_id=layer.id,
            layer=layer,
            before_array=before,
            before_image=before_image,
        ))

    def finish(
            self,
            layer: Layer | None,
            target: str,
            dirty_rect: Rect | None) -> None:
        context = self._context
        if context is None:
            return
        if self._closed or self._history_replaying():
            self._restore(context)
            self._context = None
            return
        try:
            self._finish_context(context, layer, target, dirty_rect)
        except BaseException:
            # Keep the before-state alive. The controller's failure path calls
            # on_edit_cancel, which restores it even when copying or history
            # registration failed.
            raise
        else:
            self._context = None

    def _finish_context(
            self,
            context: _EditContext,
            layer: Layer | None,
            target: str,
            dirty_rect: Rect | None) -> None:
        if target != context.target:
            self._restore(context)
            return
        if target == "selection":
            self._finish_array_edit(
                context,
                self._layer_stack.selection.data,
                dirty_rect,
                width=self._layer_stack.width,
                height=self._layer_stack.height,
            )
            return

        current = self._current_layer(context)
        if current is None or layer is not context.layer:
            self._restore(context)
            return
        if target == "transform":
            self._finish_transform(context, current)
            return
        if target == "mask_image":
            self._finish_mask_image_edit(context, current, dirty_rect)
            return
        after = current.mask.data if target == "mask" else current.image
        self._finish_array_edit(
            context,
            after,
            dirty_rect,
            width=current.width,
            height=current.height,
        )

    def cancel(
            self,
            _layer: Layer | None = None,
            _target: str = "") -> None:
        context, self._context = self._context, None
        if context is None:
            return
        self._restore(context)
        self._notify_edit_cancelled()

    def discard(self) -> None:
        """Forget a gesture after the whole document was externally replaced."""
        self._context = None

    def close(self) -> None:
        if self._closed:
            return
        errors: list[tuple[BaseException, object]] = []

        def cleanup(callback) -> None:
            try:
                callback()
            except BaseException as exc:
                errors.append((exc, exc.__traceback__))

        cleanup(self.cancel)
        self._closed = True
        remove_listener, self._remove_mutation_listener = (
            self._remove_mutation_listener, None)
        if remove_listener is not None:
            cleanup(remove_listener)
        source, callbacks = self._source, self._previous_callbacks
        self._source = None
        self._previous_callbacks = None
        if source is not None and callbacks is not None:
            if source.on_edit_begin == self.begin:
                source.on_edit_begin = callbacks[0]
            if source.on_edit_end == self.finish:
                source.on_edit_end = callbacks[1]
            if source.on_edit_cancel == self.cancel:
                source.on_edit_cancel = callbacks[2]
        if errors:
            error, traceback = errors[0]
            raise error.with_traceback(traceback)

    def _activate_context(self, context: _EditContext) -> None:
        self._context = context
        self._on_mutation_begin()

    def _cancel_source_interaction(self) -> None:
        if self._closed:
            return
        cancel = self._cancel_interaction
        if cancel is None and self._source is not None:
            cancel = getattr(self._source, "pointer_cancel", None)
            if cancel is None:
                cancel = getattr(
                    self._source, "cancel_pointer_interaction", None)
        if cancel is not None:
            cancel()
        else:
            self.cancel()

    def _finish_transform(
            self,
            context: _EditContext,
            layer: Layer) -> None:
        before = context.before_offset
        after = (layer.x, layer.y)
        if before is None or before == after:
            return
        layer_id = layer.id

        def apply_offset(offset: tuple[int, int]) -> None:
            target_layer = self._layer_stack.find_layer_by_id(layer_id)
            if target_layer is None:
                raise RuntimeError(
                    f"cannot apply canvas edit: layer {layer_id!r} is detached"
                )
            old_bounds = target_layer.bounds
            self._layer_stack.set_layer_offset(
                target_layer,
                offset[0],
                offset[1],
                old_bounds=target_layer.bounds,
                notify=False,
            )
            self._notify_changed(
                DocumentChangeKind.TRANSFORM,
                layer=target_layer,
                dirty_rect=self._layer_stack._union_rect(
                    old_bounds, target_layer.bounds),
            )

        self._document.push_callbacks(
            label=context.label,
            undo_fn=lambda: apply_offset(before),
            redo_fn=lambda: apply_offset(after),
            size_bytes=16,
        )
        self._notify_changed(
            DocumentChangeKind.TRANSFORM,
            layer=layer,
            dirty_rect=self._layer_stack._union_rect(
                (before[0], before[1], before[0] + layer.width,
                 before[1] + layer.height),
                layer.bounds,
            ),
        )
        self._notify_history_changed()

    def _finish_array_edit(
            self,
            context: _EditContext,
            after_array: np.ndarray,
            dirty_rect: Rect | None,
            *,
            width: int,
            height: int) -> None:
        rect = self._clip_rect(dirty_rect, width, height)
        before_array = context.before_array
        if rect is None or before_array is None:
            return
        x0, y0, x1, y1 = rect
        before = before_array[y0:y1, x0:x1].copy()
        after = after_array[y0:y1, x0:x1].copy()
        if np.array_equal(before, after):
            return
        layer_id = context.layer_id
        target = context.target

        self._document.push_callbacks(
            label=context.label,
            undo_fn=lambda: self._apply_patch(layer_id, target, rect, before),
            redo_fn=lambda: self._apply_patch(layer_id, target, rect, after),
            size_bytes=before.nbytes + after.nbytes,
        )
        layer = (
            self._layer_stack.find_layer_by_id(layer_id)
            if layer_id is not None else None
        )
        if target == "image" and layer is not None:
            self._notify_changed(
                DocumentChangeKind.PIXELS, layer=layer, dirty_rect=rect)
        else:
            self._notify_changed(
                DocumentChangeKind.METADATA, layer=layer)
        self._notify_history_changed()

    def _finish_mask_image_edit(
            self,
            context: _EditContext,
            layer: Layer,
            dirty_rect: Rect | None) -> None:
        rect = self._clip_rect(dirty_rect, layer.width, layer.height)
        before_mask = context.before_array
        before_image = context.before_image
        if rect is None or before_mask is None or before_image is None:
            return
        x0, y0, x1, y1 = rect
        before_mask_patch = before_mask[y0:y1, x0:x1].copy()
        after_mask_patch = layer.mask.data[y0:y1, x0:x1].copy()
        before_image_patch = before_image[y0:y1, x0:x1].copy()
        after_image_patch = layer.image[y0:y1, x0:x1].copy()
        if (
                np.array_equal(before_mask_patch, after_mask_patch)
                and np.array_equal(before_image_patch, after_image_patch)):
            return
        layer_id = context.layer_id
        if layer_id is None:
            return

        self._document.push_callbacks(
            label=context.label,
            undo_fn=lambda: self._apply_mask_image_patch(
                layer_id,
                rect,
                before_mask_patch,
                before_image_patch,
            ),
            redo_fn=lambda: self._apply_mask_image_patch(
                layer_id,
                rect,
                after_mask_patch,
                after_image_patch,
            ),
            size_bytes=(
                before_mask_patch.nbytes
                + after_mask_patch.nbytes
                + before_image_patch.nbytes
                + after_image_patch.nbytes
            ),
        )
        self._notify_changed(
            DocumentChangeKind.PIXELS, layer=layer, dirty_rect=rect)
        self._notify_history_changed()

    def _apply_mask_image_patch(
            self,
            layer_id: str,
            rect: Rect,
            mask_patch: np.ndarray,
            image_patch: np.ndarray) -> None:
        layer = self._layer_stack.find_layer_by_id(layer_id)
        if layer is None:
            raise RuntimeError(
                f"cannot apply canvas edit: layer {layer_id!r} is detached"
            )
        clipped = self._clip_rect(rect, layer.width, layer.height)
        x0, y0, x1, y1 = rect
        if (
                clipped != rect
                or layer.mask.data[y0:y1, x0:x1].shape != mask_patch.shape
                or layer.image[y0:y1, x0:x1].shape != image_patch.shape):
            raise RuntimeError("layer geometry changed since canvas edit")
        layer.mask.data[y0:y1, x0:x1] = coerce_mask_data(mask_patch)
        layer.image[y0:y1, x0:x1] = image_patch
        self._layer_stack.mark_layer_dirty(
            layer,
            rect=layer.local_rect_to_canvas(rect),
        )
        self._notify_changed(
            DocumentChangeKind.PIXELS, layer=layer, dirty_rect=rect)

    def _apply_patch(
            self,
            layer_id: str | None,
            target: str,
            rect: Rect,
            patch: np.ndarray) -> None:
        x0, y0, x1, y1 = rect
        if target == "selection":
            destination = self._layer_stack.selection.data
            clipped = self._clip_rect(
                rect,
                self._layer_stack.width,
                self._layer_stack.height,
            )
            if clipped != rect or destination[y0:y1, x0:x1].shape != patch.shape:
                raise RuntimeError("selection geometry changed since canvas edit")
            destination[y0:y1, x0:x1] = coerce_mask_data(patch)
            self._notify_changed(DocumentChangeKind.METADATA)
            return

        layer = (
            self._layer_stack.find_layer_by_id(layer_id)
            if layer_id is not None else None
        )
        if layer is None:
            raise RuntimeError(
                f"cannot apply canvas edit: layer {layer_id!r} is detached"
            )
        clipped = self._clip_rect(rect, layer.width, layer.height)
        if clipped != rect:
            raise RuntimeError("layer geometry changed since canvas edit")
        if target == "mask":
            if layer.mask.data[y0:y1, x0:x1].shape != patch.shape:
                raise RuntimeError("layer mask geometry changed since canvas edit")
            layer.mask.data[y0:y1, x0:x1] = coerce_mask_data(patch)
            self._notify_changed(
                DocumentChangeKind.METADATA, layer=layer)
            return
        if layer.image[y0:y1, x0:x1].shape != patch.shape:
            raise RuntimeError("layer image geometry changed since canvas edit")
        layer.image[y0:y1, x0:x1] = patch
        self._layer_stack.mark_layer_dirty(
            layer,
            rect=layer.local_rect_to_canvas(rect),
        )
        self._notify_changed(
            DocumentChangeKind.PIXELS, layer=layer, dirty_rect=rect)

    def _restore(self, context: _EditContext) -> None:
        if context.target == "selection":
            before = context.before_array
            if (
                    before is not None
                    and before.shape == self._layer_stack.selection.data.shape):
                self._layer_stack.selection.data[:] = before
                self._notify_changed(DocumentChangeKind.METADATA)
            return

        layer = self._current_layer(context)
        if layer is None:
            layer = context.layer
        if layer is None:
            return
        if context.target == "transform":
            before = context.before_offset
            if before is not None:
                if self._current_layer(context) is layer:
                    old_bounds = layer.bounds
                    self._layer_stack.set_layer_offset(
                        layer,
                        before[0],
                        before[1],
                        old_bounds=layer.bounds,
                        notify=False,
                    )
                    self._notify_changed(
                        DocumentChangeKind.TRANSFORM,
                        layer=layer,
                        dirty_rect=self._layer_stack._union_rect(
                            old_bounds, layer.bounds),
                    )
                else:
                    # The orphan is no longer part of the aggregate, but its
                    # in-flight tool mutation must still be rolled back.
                    layer.x, layer.y = before
            return
        before = context.before_array
        if before is None:
            return
        if context.target == "mask" and before.shape == layer.mask.data.shape:
            layer.mask.data[:] = before
            self._notify_changed(
                DocumentChangeKind.METADATA, layer=layer)
        elif (
                context.target == "mask_image"
                and context.before_image is not None
                and before.shape == layer.mask.data.shape
                and context.before_image.shape == layer.image.shape):
            layer.mask.data[:] = before
            layer.image[:] = context.before_image
            self._layer_stack.mark_layer_dirty(layer, rect=layer.bounds)
            self._notify_changed(
                DocumentChangeKind.PIXELS,
                layer=layer,
                dirty_rect=(0, 0, layer.width, layer.height),
            )
        elif context.target == "image" and before.shape == layer.image.shape:
            layer.image[:] = before
            self._layer_stack.mark_layer_dirty(
                layer,
                rect=layer.bounds,
            )
            self._notify_changed(
                DocumentChangeKind.PIXELS,
                layer=layer,
                dirty_rect=(0, 0, layer.width, layer.height),
            )

    def _current_layer(self, context: _EditContext) -> Layer | None:
        if context.layer_id is None:
            return None
        return self._layer_stack.find_layer_by_id(context.layer_id)

    def _notify_changed(
            self,
            kind: DocumentChangeKind,
            *,
            layer: Layer | None = None,
            dirty_rect: Rect | None = None) -> None:
        self._layer_stack.publish_change(
            kind,
            layers=(layer,) if layer is not None else (),
            dirty_rect=dirty_rect,
            operation="canvas edit",
        )

    def _notify_history_changed(self) -> None:
        try:
            self._on_history_changed()
        except Exception:
            logger.exception(
                "Canvas history observer failed after a completed mutation")

    def _notify_edit_cancelled(self) -> None:
        try:
            self._on_edit_cancelled()
        except Exception:
            logger.exception(
                "Canvas observer failed after a cancelled mutation")

    @staticmethod
    def _clip_rect(
            rect: Rect | None,
            width: int,
            height: int) -> Rect | None:
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        x0 = max(0, min(int(x0), width))
        y0 = max(0, min(int(y0), height))
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
