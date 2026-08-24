"""Small, dependency-free helpers for canonical-head checkpoint retention."""

from __future__ import annotations


def periodic_epoch_checkpoint_due(
    epoch: int,
    *,
    full_epoch: bool,
    every_epochs: int,
) -> bool:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if every_epochs < 0:
        raise ValueError("every_epochs must be non-negative")
    return bool(full_epoch and every_epochs and epoch % every_epochs == 0)
