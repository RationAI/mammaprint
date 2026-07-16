from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
import torch


class LabelMode(Enum):
    TYPE = "type"
    INDEX = "index"


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

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    """One slide's target as a ``(1,)`` float tensor.

    Both modes are single-output (``out_dim=1``): binary classification (luminal
    a=1 / b=0, for ``BCEWithLogitsLoss`` + binary metrics) and regression (the
    MammaPrint index). Shape ``(1,)`` matches the head's per-slide output so batched
    preds/targets line up as ``(B, 1)``.
    """
    match mode:
        case LabelMode.TYPE:
            value = float(slide_metadata["type_label"])
        case LabelMode.INDEX:
            value = float(slide_metadata["mammaprint_index"])
        case _:
            raise ValueError(f"Unsupported label mode: {mode}")

    return torch.tensor([value], dtype=torch.float32)


def get_target_column(mode: LabelMode) -> str:
    match mode:
        case LabelMode.TYPE:
            return "type_label"
        case LabelMode.INDEX:
            return "mammaprint_index"

    raise ValueError(f"Unsupported label mode: {mode}")
