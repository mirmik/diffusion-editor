from __future__ import annotations

import runpy
from pathlib import Path


MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "experiment-qwen-front-replicas.py")
)


def test_replica_seeds_are_independent_and_reproducible() -> None:
    assert MODULE["replica_seeds"](100, 4) == [100, 101, 102, 103]
