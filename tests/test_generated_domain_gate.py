from __future__ import annotations

import unittest

from diffusion_editor.training.checkpointing import periodic_epoch_checkpoint_due
from diffusion_editor.training.generated_domain_gate import (
    METRIC_DIRECTIONS,
    rank_generated_domain_candidates,
)


def _metrics(value: float, *, higher: bool = False) -> dict[str, float]:
    return {
        metric: (-value if direction == "max" and not higher else value)
        for metric, direction in METRIC_DIRECTIONS.items()
    }


class GeneratedDomainGateTest(unittest.TestCase):
    def test_periodic_checkpoint_requires_matching_full_epoch(self) -> None:
        self.assertTrue(
            periodic_epoch_checkpoint_due(4, full_epoch=True, every_epochs=2)
        )
        self.assertFalse(
            periodic_epoch_checkpoint_due(3, full_epoch=True, every_epochs=2)
        )
        self.assertFalse(
            periodic_epoch_checkpoint_due(4, full_epoch=False, every_epochs=2)
        )
        self.assertFalse(
            periodic_epoch_checkpoint_due(4, full_epoch=True, every_epochs=0)
        )

    def test_equal_identity_rank_selects_balanced_candidate(self) -> None:
        candidates = {
            "balanced": {
                "jay": _metrics(1.0),
                "victoria": _metrics(1.0),
            },
            "jay-only": {
                "jay": _metrics(0.0),
                "victoria": _metrics(2.0),
            },
            "victoria-only": {
                "jay": _metrics(2.0),
                "victoria": _metrics(0.0),
            },
        }

        result = rank_generated_domain_candidates(candidates)

        self.assertEqual(result["selected_candidate"], "balanced")
        self.assertIn("balanced", result["pareto_candidates"])

    def test_maximize_metrics_are_ranked_in_reverse(self) -> None:
        low = _metrics(1.0)
        high = dict(low)
        high["image_foreground_iou_mean"] = 2.0
        high["voxel_fraction_at_least_2_views"] = 2.0
        candidates = {
            "low": {"jay": low},
            "high": {"jay": high},
        }

        result = rank_generated_domain_candidates(candidates)

        self.assertLess(result["scores"]["high"], result["scores"]["low"])

    def test_rejects_identity_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same order"):
            rank_generated_domain_candidates(
                {
                    "first": {"jay": _metrics(1.0)},
                    "second": {"victoria": _metrics(1.0)},
                }
            )


if __name__ == "__main__":
    unittest.main()
