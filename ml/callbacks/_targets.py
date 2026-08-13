"""Shared output semantics for prediction callbacks."""

from dataclasses import dataclass

from ml.data.datasets.labels import LabelMode


@dataclass(frozen=True)
class PredictionTarget:
    """One model output and its pathologist-facing interpretation."""

    name: str
    output_index: int
    is_classification: bool


def prediction_targets(label_mode: str | LabelMode) -> tuple[PredictionTarget, ...]:
    """Return the ordered output specification for a configured task."""
    mode = label_mode if isinstance(label_mode, LabelMode) else LabelMode(label_mode)
    if mode is LabelMode.TYPE:
        return (PredictionTarget("luminal_a_probability", 0, True),)
    if mode is LabelMode.INDEX:
        return (PredictionTarget("mammaprint_index", 0, False),)
    if mode is LabelMode.BOTH:
        return (
            PredictionTarget("luminal_a_probability", 0, True),
            PredictionTarget("mammaprint_index", 1, False),
        )
    raise ValueError(f"Unsupported label mode: {mode}")


__all__ = ["PredictionTarget", "prediction_targets"]
