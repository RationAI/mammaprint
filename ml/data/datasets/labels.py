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
            slides["index"] = slides["index"].astype(float)

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides


def get_label(slide_metadata: pd.Series, mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.TYPE:
            return torch.tensor(int(slide_metadata["type_label"])).long()
        case LabelMode.INDEX:
            return torch.tensor(float(slide_metadata["index"])).float()

    raise ValueError(f"Unsupported label mode: {mode}")


def get_target_column(mode: LabelMode) -> str:
    match mode:
        case LabelMode.TYPE:
            return "type_label"
        case LabelMode.INDEX:
            return "index"

    raise ValueError(f"Unsupported label mode: {mode}")
