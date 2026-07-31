"""Typed, revisioned notifications for persistent document changes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable, Protocol
from weakref import ref


Rect = tuple[int, int, int, int]


class DocumentChangeKind(str, Enum):
    PIXELS = "pixels"
    TRANSFORM = "transform"
    STRUCTURE = "structure"
    VISIBILITY = "visibility"
    OPACITY = "opacity"
    ACTIVE = "active"
    METADATA = "metadata"
    SNAPSHOT_RESTORE = "snapshot_restore"


@dataclass(frozen=True)
class DocumentChangeEvent:
    """One committed semantic document mutation.

    ``dirty_rect`` is layer-local for pixel events and canvas-local for
    transform events.  It is always clipped to the corresponding bounds.
    """

    kind: DocumentChangeKind
    revision: int
    layer_ids: tuple[str, ...] = ()
    dirty_rect: Rect | None = None


DocumentChangeCallback = Callable[[DocumentChangeEvent], None]


class _SubscriptionOwner(Protocol):
    def _unsubscribe_change(self, token: int) -> None: ...


class DocumentChangeSubscription:
    """Idempotent ownership handle returned by ``LayerStack.subscribe``."""

    def __init__(self, owner: _SubscriptionOwner, token: int) -> None:
        self._owner = ref(owner)
        self._token = token
        self._lock = RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def unsubscribe(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            owner = self._owner()
        if owner is not None:
            owner._unsubscribe_change(self._token)

    close = unsubscribe

    def __enter__(self) -> "DocumentChangeSubscription":
        return self

    def __exit__(self, *_args: object) -> None:
        self.unsubscribe()
