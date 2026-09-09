"""Shared SDXL prediction-mode detection for the editor and Studio."""
from pathlib import Path

VPRED_HINTS = ("vpred", "v-pred", "v_pred", "vprediction", "v-prediction", "v_prediction")


def guess_prediction_type(model_path):
    name = Path(model_path).name.lower()
    return "v_prediction" if any(hint in name for hint in VPRED_HINTS) else None


def resolve_prediction_type(model_path, override=None):
    if override not in (None, "", "auto", "epsilon", "v_prediction"):
        raise ValueError(f"Unsupported prediction mode: {override}")
    return override if override in ("epsilon", "v_prediction") else guess_prediction_type(model_path) or "epsilon"
