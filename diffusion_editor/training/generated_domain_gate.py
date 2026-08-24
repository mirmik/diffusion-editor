"""Checkpoint ranking for generated-domain canonical point-map validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


METRIC_DIRECTIONS = {
    "surface_distance_median": "min",
    "surface_distance_p95": "min",
    "surface_coverage_p95": "min",
    "projective_reprojection_median_mean": "min",
    "projective_reprojection_p95_max": "min",
    "image_foreground_iou_mean": "max",
    "voxel_fraction_at_least_2_views": "max",
}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def rank_generated_domain_candidates(
    candidates: Mapping[str, Mapping[str, Mapping[str, float]]],
    candidate_order: Sequence[str] | None = None,
) -> dict:
    """Rank candidates equally across identities and declared metrics.

    Every identity/metric pair contributes one normalized rank in [0, 1]. This
    avoids allowing an identity with a larger numeric scale or more pixels to
    dominate checkpoint selection.
    """
    names = list(candidate_order or candidates)
    if set(names) != set(candidates):
        raise ValueError("candidate_order must contain every candidate exactly once")
    if not names:
        raise ValueError("at least one candidate is required")
    identities = list(candidates[names[0]])
    if not identities:
        raise ValueError("at least one validation identity is required")
    for name in names:
        if list(candidates[name]) != identities:
            raise ValueError("all candidates must contain identities in the same order")

    contributions = {name: [] for name in names}
    rank_table = {name: {} for name in names}
    denominator = max(len(names) - 1, 1)
    for identity in identities:
        for metric, direction in METRIC_DIRECTIONS.items():
            values = np.asarray(
                [float(candidates[name][identity][metric]) for name in names],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite {identity}/{metric}")
            ranked_values = values if direction == "min" else -values
            ranks = _average_ranks(ranked_values)
            for name, rank in zip(names, ranks):
                normalized = float((rank - 1.0) / denominator)
                contributions[name].append(normalized)
                rank_table[name][f"{identity}/{metric}"] = normalized

    scores = {
        name: float(np.mean(contributions[name]))
        for name in names
    }

    def dominates(first: str, second: str) -> bool:
        first_values = []
        second_values = []
        for identity in identities:
            for metric, direction in METRIC_DIRECTIONS.items():
                a = float(candidates[first][identity][metric])
                b = float(candidates[second][identity][metric])
                first_values.append(a if direction == "min" else -a)
                second_values.append(b if direction == "min" else -b)
        return all(a <= b for a, b in zip(first_values, second_values)) and any(
            a < b for a, b in zip(first_values, second_values)
        )

    pareto = [
        name
        for name in names
        if not any(dominates(other, name) for other in names if other != name)
    ]
    selected = min(names, key=lambda name: (scores[name], names.index(name)))
    return {
        "selection_rule": (
            "Lowest mean normalized rank over every validation identity and the "
            "seven declared metrics; identities and metrics have equal weight. "
            "Ties follow declared candidate order."
        ),
        "metric_directions": dict(METRIC_DIRECTIONS),
        "scores": scores,
        "rank_contributions": rank_table,
        "pareto_candidates": pareto,
        "selected_candidate": selected,
    }
