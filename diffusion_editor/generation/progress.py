"""Provider-neutral progress for one stage of an active generation job."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationProgress:
    message: str
    current: int | None = None
    total: int | None = None

    def __post_init__(self):
        if not isinstance(self.message, str):
            raise ValueError("Progress message must be a string")
        if self.current is None and self.total is None:
            return
        if (type(self.current) is not int or type(self.total) is not int
                or self.total <= 0 or not 0 <= self.current <= self.total):
            raise ValueError("Progress must satisfy 0 <= current <= total, total > 0")

    @property
    def fraction(self) -> float | None:
        return self.current / self.total if self.total is not None else None

    @property
    def text(self) -> str:
        if self.total is None:
            return self.message
        return f"{self.message}: {self.current}/{self.total} ({self.fraction:.0%})"

    def to_dict(self) -> dict:
        return {"message": self.message, "current": self.current, "total": self.total}

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(payload.get("message"), payload.get("current"), payload.get("total"))
