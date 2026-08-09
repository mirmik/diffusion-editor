"""Document identity, dirty tracking and bounded crash recovery storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable
import uuid

from .change_event import DocumentChangeEvent


def _digest(snapshot: bytes) -> str:
    return hashlib.sha256(snapshot).hexdigest()


@dataclass(frozen=True)
class RecoveryRecord:
    session_id: str
    created_at: float
    project_path: str | None
    snapshot_path: Path
    snapshot_bytes: int


class DocumentSession:
    """Toolkit-neutral identity and saved/dirty state for one document."""

    def __init__(self) -> None:
        self.session_id = self._new_id()
        self.revision = 0
        self.saved_revision = 0
        self.path: str | None = None
        self._saved_digest: str | None = None
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def record_change(self, event: DocumentChangeEvent) -> None:
        self.revision = max(self.revision, event.revision)
        self._dirty = True

    def mark_external_mutation(self) -> int:
        self.revision += 1
        self._dirty = True
        return self.revision

    def reset(
            self,
            *,
            revision: int,
            path: str | None,
            snapshot: bytes,
            clean: bool = True) -> None:
        self.session_id = self._new_id()
        self.revision = revision
        self.path = path
        self._saved_digest = _digest(snapshot) if clean else None
        self.saved_revision = revision if clean else -1
        self._dirty = not clean

    def mark_saved(self, path: str, snapshot: bytes) -> None:
        self.path = path
        self.saved_revision = self.revision
        self._saved_digest = _digest(snapshot)
        self._dirty = False

    def mark_discarded(self) -> None:
        self.saved_revision = self.revision
        self._saved_digest = None
        self._dirty = False

    def reconcile(self, snapshot: bytes) -> bool:
        """Recompute dirty after undo/redo may have returned to saved bytes."""
        clean = (
            self._saved_digest is not None
            and _digest(snapshot) == self._saved_digest
        )
        self._dirty = not clean
        if clean:
            self.saved_revision = self.revision
        return clean

    @staticmethod
    def _new_id() -> str:
        return f"document_{uuid.uuid4().hex}"


class RecoveryStore:
    """Atomic recovery projects left behind only by dirty sessions."""

    def __init__(
            self,
            root: str | os.PathLike[str],
            *,
            max_snapshot_bytes: int | None = None,
            max_records: int = 3) -> None:
        if (
                (max_snapshot_bytes is not None and max_snapshot_bytes <= 0)
                or max_records <= 0):
            raise ValueError("recovery limits must be positive")
        self.root = Path(root)
        self.max_snapshot_bytes = (
            int(max_snapshot_bytes) if max_snapshot_bytes is not None else None)
        self.max_records = int(max_records)

    def write(self, session: DocumentSession, snapshot: bytes) -> RecoveryRecord:
        self._validate_snapshot_size(len(snapshot))
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.root / f"{session.session_id}.deproj"
        self._atomic_write(snapshot_path, snapshot)
        return self._commit_record(session, snapshot_path)

    def write_project(
            self,
            session: DocumentSession,
            writer: Callable[[str], None]) -> RecoveryRecord:
        """Write a compressed project directly to disk without a RAM snapshot."""
        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.root / f"{session.session_id}.deproj"
        staged_path = self.root / f".{session.session_id}.pending.deproj"
        try:
            writer(str(staged_path))
            self._validate_snapshot_size(staged_path.stat().st_size)
            os.replace(staged_path, snapshot_path)
        except BaseException:
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return self._commit_record(session, snapshot_path)

    def _commit_record(
            self,
            session: DocumentSession,
            snapshot_path: Path) -> RecoveryRecord:
        snapshot_bytes = snapshot_path.stat().st_size
        metadata_path = self.root / f"{session.session_id}.json"
        created_at = time.time()
        metadata = json.dumps({
            "session_id": session.session_id,
            "created_at": created_at,
            "project_path": session.path,
            "snapshot_bytes": snapshot_bytes,
        }, sort_keys=True).encode("utf-8")
        self._atomic_write(metadata_path, metadata)
        self._prune()
        return RecoveryRecord(
            session_id=session.session_id,
            created_at=created_at,
            project_path=session.path,
            snapshot_path=snapshot_path,
            snapshot_bytes=snapshot_bytes,
        )

    def records(self) -> tuple[RecoveryRecord, ...]:
        if not self.root.is_dir():
            return ()
        records: list[RecoveryRecord] = []
        for metadata_path in self.root.glob("document_*.json"):
            try:
                payload = json.loads(metadata_path.read_text("utf-8"))
                session_id = str(payload["session_id"])
                snapshot_path = self.root / f"{session_id}.deproj"
                snapshot_bytes = int(payload["snapshot_bytes"])
                if (
                        not snapshot_path.is_file()
                        or snapshot_bytes != snapshot_path.stat().st_size
                        or (
                            self.max_snapshot_bytes is not None
                            and snapshot_bytes > self.max_snapshot_bytes)):
                    continue
                records.append(RecoveryRecord(
                    session_id=session_id,
                    created_at=float(payload["created_at"]),
                    project_path=(
                        str(payload["project_path"])
                        if payload.get("project_path") else None),
                    snapshot_path=snapshot_path,
                    snapshot_bytes=snapshot_bytes,
                ))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
        return tuple(sorted(
            records, key=lambda record: record.created_at, reverse=True))

    def latest(self) -> RecoveryRecord | None:
        records = self.records()
        return records[0] if records else None

    def load(self, record: RecoveryRecord) -> bytes:
        snapshot = record.snapshot_path.read_bytes()
        if (
                len(snapshot) != record.snapshot_bytes
                or (
                    self.max_snapshot_bytes is not None
                    and len(snapshot) > self.max_snapshot_bytes)):
            raise ValueError("recovery snapshot is incomplete or oversized")
        return snapshot

    def _validate_snapshot_size(self, snapshot_bytes: int) -> None:
        if (
                self.max_snapshot_bytes is not None
                and snapshot_bytes > self.max_snapshot_bytes):
            raise ValueError("recovery snapshot exceeds configured size limit")

    def discard(self, session_id: str) -> None:
        for suffix in (".json", ".deproj"):
            try:
                (self.root / f"{session_id}{suffix}").unlink()
            except FileNotFoundError:
                pass

    def _prune(self) -> None:
        for record in self.records()[self.max_records:]:
            self.discard(record.session_id)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent, delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
