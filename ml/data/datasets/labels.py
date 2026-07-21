from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
import torch


class LabelMode(Enum):
    TYPE = "type"
    INDEX = "index"
    BOTH = "both"


def _map_luminal_type(value: object) -> int:
    if value == "a luminal":
        return 1
    if value == "b luminal":
        return 0
    raise ValueError(f"Unknown luminal type label: {value!r}")


def process_slides(slides: pd.DataFrame, mode: LabelMode | None = None) -> pd.DataFrame:
    match mode:
        case LabelMode.TYPE:
            slides = slides.copy()
            slides["type_label"] = slides["type"].map(_map_luminal_type)
        case LabelMode.INDEX:
            slides = slides.copy()
            slides["mammaprint_index"] = slides["mammaprint_index"].astype(float)
        case LabelMode.BOTH:
            slides = slides.copy()
            slides["type_label"] = slides["type"].map(_map_luminal_type)
            slides["mammaprint_index"] = slides["mammaprint_index"].astype(float)

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    """One slide's target as a float tensor.

    The single-task modes are single-output (``out_dim=1``): binary classification
    (luminal a=1 / b=0, for ``BCEWithLogitsLoss`` + binary metrics) and regression
    (the MammaPrint index). Shape ``(1,)`` matches the head's per-slide output so
    batched preds/targets line up as ``(B, 1)``.

    ``BOTH`` is the multi-task target: ``[type_label, mammaprint_index]`` as a
    ``(2,)`` tensor, column 0 the class and column 1 the index. Pair it with a
    2-output head and :class:`~ml.models.losses.joint.JointLoss`, which slices the
    columns back apart.
    """
    match mode:
        case LabelMode.TYPE:
            values = [float(slide_metadata["type_label"])]
        case LabelMode.INDEX:
            values = [float(slide_metadata["mammaprint_index"])]
        case LabelMode.BOTH:
            values = [
                float(slide_metadata["type_label"]),
                float(slide_metadata["mammaprint_index"]),
            ]
        case _:
            raise ValueError(f"Unsupported label mode: {mode}")

    return torch.tensor(values, dtype=torch.float32)


def get_target_columns(mode: LabelMode) -> list[str]:
    """Label column(s) that must be present for a slide to be usable in ``mode``.

    ``BOTH`` requires both columns, so a slide missing either is dropped.
    """
    match mode:
        case LabelMode.TYPE:
            return ["type_label"]
        case LabelMode.INDEX:
            return ["mammaprint_index"]
        case LabelMode.BOTH:
            return ["type_label", "mammaprint_index"]

    raise ValueError(f"Unsupported label mode: {mode}")
