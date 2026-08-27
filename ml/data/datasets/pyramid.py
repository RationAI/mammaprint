"""Lazy alignment of existing per-level embedding artifacts into pyramids."""

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ml.data.datasets._sources import (
    available_slides,
    download_level_sources,
    load_labeled_slides,
    split_uri,
)
from ml.data.datasets.labels import LabelMode, get_label
from ml.typing import (
    Coordinate,
    LevelSpec,
    OverlapPolicy,
    PyramidBag,
    PyramidSample,
    PyramidSelectionAudit,
    PyramidSlideMetadata,
)


logger = logging.getLogger(__name__)


class PyramidDiagnostics(NamedTuple):
    """Counts recorded while complete anchors are selected for one slide."""

    candidates: int
    complete: int
    missing_by_level: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class PyramidLayout:
    """Validated level ordering and coordinate arithmetic for one pyramid shape."""

    anchor_level: int
    level_order: tuple[int, ...]
    ratios: tuple[int, ...]
    tile_extent: int

    @classmethod
    def from_levels(
        cls,
        levels: Mapping[int, LevelSpec] | Mapping[str, LevelSpec],
        *,
        anchor_level: int,
        level_order: Sequence[int],
        tile_extent: int,
    ) -> "PyramidLayout":
        """Validate level cards and derive integer coarse-to-fine scale ratios."""
        normalized_order = tuple(int(level) for level in level_order)
        if tile_extent <= 0:
            raise ValueError(f"tile_extent must be positive; got {tile_extent}.")
        if not normalized_order or normalized_order[0] != anchor_level:
            raise ValueError("level_order must start with anchor_level.")
        if len(set(normalized_order)) != len(normalized_order):
            raise ValueError("level_order must not contain duplicate levels.")

        cards = {int(level): card for level, card in levels.items()}
        missing = [level for level in normalized_order if level not in cards]
        if missing:
            raise ValueError(f"Missing level cards for levels {missing}.")

        anchor_mpp = float(cards[anchor_level]["mpp"])
        if anchor_mpp <= 0:
            raise ValueError(f"Level {anchor_level} mpp must be positive.")

        ratios: list[int] = []
        for level in normalized_order:
            card = cards[level]
            card_extent = int(card["tile_extent"])
            if card_extent != tile_extent:
                raise ValueError(
                    f"Level {level} tile_extent is {card_extent}, expected "
                    f"configured tile_extent {tile_extent}."
                )
            mpp = float(card["mpp"])
            if mpp <= 0:
                raise ValueError(f"Level {level} mpp must be positive.")
            raw_ratio = anchor_mpp / mpp
            ratio = round(raw_ratio)
            if ratio < 1 or not math.isclose(
                raw_ratio, ratio, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(
                    f"Level {level} must have a positive integer scale ratio to "
                    f"anchor level {anchor_level}; got {raw_ratio}."
                )
            ratios.append(ratio)

        return cls(anchor_level, normalized_order, tuple(ratios), tile_extent)

    @property
    def token_count(self) -> int:
        """Number of ordered tile tokens in each complete pyramid."""
        return sum(ratio * ratio for ratio in self.ratios)

    def coordinates_by_level(
        self, anchor: Coordinate
    ) -> dict[int, tuple[Coordinate, ...]]:
        """Return required coordinates, coarse-to-fine and row-major per level."""
        x, y = anchor
        return {
            level: tuple(
                (
                    ratio * x + column * self.tile_extent,
                    ratio * y + row * self.tile_extent,
                )
                for row in range(ratio)
                for column in range(ratio)
            )
            for level, ratio in zip(self.level_order, self.ratios, strict=True)
        }

    def coordinates(self, anchor: Coordinate) -> tuple[Coordinate, ...]:
        """Flatten all required coordinates in deterministic token order."""
        return tuple(
            coordinate
            for coordinates in self.coordinates_by_level(anchor).values()
            for coordinate in coordinates
        )


class PyramidScaleDataset(Dataset[PyramidSample]):
    """Lazily gather complete aligned pyramids from existing embedding Parquets."""

    def __init__(
        self,
        split: str,
        levels: Mapping[str | int, LevelSpec],
        data_mapping: str | Path,
        anchor_level: int,
        level_order: Sequence[int],
        tile_extent: int,
        label_mode: str = "type",
        require_complete: bool = True,
        overlap_policy: OverlapPolicy | str = OverlapPolicy.NON_OVERLAPPING,
        selection_seed: int = 0,
    ) -> None:
        if not require_complete:
            raise ValueError(
                "require_complete=false is not supported; pyramid samples must be complete."
            )

        self.label_mode = LabelMode(label_mode)
        self.split = split
        self.overlap_policy = OverlapPolicy(overlap_policy)
        self.selection_seed = selection_seed
        self.levels = {int(level): card for level, card in levels.items()}
        self.layout = PyramidLayout.from_levels(
            self.levels,
            anchor_level=anchor_level,
            level_order=level_order,
            tile_extent=tile_extent,
        )
        sources = {
            level: split_uri(self.levels[level], "uris", split, level)
            for level in self.layout.level_order
        }
        self.embeddings_dirs = download_level_sources(sources)

        slides = load_labeled_slides(data_mapping, self.label_mode)
        present = available_slides(self.embeddings_dirs)
        slides = slides[slides["name"].isin(present)].reset_index(drop=True)

        self.anchors_by_slide: dict[str, tuple[Coordinate, ...]] = {}
        self.diagnostics: dict[str, PyramidDiagnostics] = {}
        audit_records: list[PyramidSelectionAudit] = []
        for _, row in slides.iterrows():
            name = str(row["name"])
            coordinates = self._read_coordinate_sets(name)
            anchors = self._complete_anchors(name, coordinates)
            diagnostics = self.diagnostics[name]
            audit_records.append(
                PyramidSelectionAudit(
                    slide_id=name,
                    split=split,
                    label=self._mapping_label(row),
                    candidates=diagnostics.candidates,
                    complete_before_thinning=diagnostics.complete,
                    retained=len(anchors),
                    incomplete=diagnostics.candidates - diagnostics.complete,
                    missing_by_level=diagnostics.missing_by_level,
                    completion_rate=(
                        diagnostics.complete / diagnostics.candidates
                        if diagnostics.candidates
                        else 0.0
                    ),
                    overlap_retention_rate=(
                        len(anchors) / diagnostics.complete
                        if diagnostics.complete
                        else 0.0
                    ),
                    rejected=not anchors,
                )
            )
            if anchors:
                self.anchors_by_slide[name] = anchors

        self.audit_records: tuple[PyramidSelectionAudit, ...] = tuple(audit_records)

        self.slides = slides[slides["name"].isin(self.anchors_by_slide)].reset_index(
            drop=True
        )
        rejected = len(slides) - len(self.slides)
        retained = sum(item.retained for item in self.audit_records)
        candidates = sum(item.candidates for item in self.audit_records)
        logger.info(
            "Pyramid scan retained %d/%d anchors across %d slides; rejected %d "
            "slides with no complete pyramid.",
            retained,
            candidates,
            len(self.slides),
            rejected,
        )

    def _parquet_path(self, level: int, slide_name: str) -> Path:
        return (self.embeddings_dirs[level] / slide_name).with_suffix(".parquet")

    def _read_coordinate_sets(self, slide_name: str) -> dict[int, set[Coordinate]]:
        result: dict[int, set[Coordinate]] = {}
        for level in self.layout.level_order:
            frame = pd.read_parquet(
                self._parquet_path(level, slide_name), columns=["x", "y"]
            )
            points = tuple(
                (int(x), int(y))
                for x, y in frame[["x", "y"]].itertuples(index=False, name=None)
            )
            point_set = set(points)
            if len(point_set) != len(points):
                raise ValueError(
                    f"Slide {slide_name!r} level {level} contains a duplicate "
                    "coordinate."
                )
            result[level] = point_set
        return result

    def _mapping_label(self, row: pd.Series) -> str:
        """Return a stable source-row label without inspecting a target tensor."""
        match self.label_mode:
            case LabelMode.TYPE:
                return str(row["type"])
            case LabelMode.INDEX:
                return str(row["mammaprint_index"])
            case LabelMode.BOTH:
                return f"{row['type']}|{row['mammaprint_index']}"

    def audit_frame(self) -> pd.DataFrame:
        """Return the per-slide selection audit in mapping-row order."""
        columns = [field.name for field in fields(PyramidSelectionAudit)]
        return pd.DataFrame(
            [asdict(record) for record in self.audit_records], columns=columns
        )

    def audit_summary(self) -> pd.DataFrame:
        """Aggregate selection audit totals and retained-region quantiles by cohort."""
        columns = [
            "split",
            "label",
            "slides",
            "rejected_slides",
            "candidates",
            "complete_before_thinning",
            "retained",
            "incomplete",
            "completion_rate",
            "overlap_retention_rate",
            "retained_p25",
            "retained_p50",
            "retained_p75",
        ]
        frame = self.audit_frame()
        if frame.empty:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, str | int | float]] = []
        for (split, label), group in frame.groupby(["split", "label"], sort=True):
            candidates = int(group["candidates"].sum())
            complete = int(group["complete_before_thinning"].sum())
            retained = int(group["retained"].sum())
            rows.append(
                {
                    "split": str(split),
                    "label": str(label),
                    "slides": len(group),
                    "rejected_slides": int(group["rejected"].sum()),
                    "candidates": candidates,
                    "complete_before_thinning": complete,
                    "retained": retained,
                    "incomplete": int(group["incomplete"].sum()),
                    "completion_rate": complete / candidates if candidates else 0.0,
                    "overlap_retention_rate": retained / complete if complete else 0.0,
                    "retained_p25": float(group["retained"].quantile(0.25)),
                    "retained_p50": float(group["retained"].quantile(0.50)),
                    "retained_p75": float(group["retained"].quantile(0.75)),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def _complete_anchors(
        self,
        slide_name: str,
        coordinates: Mapping[int, set[Coordinate]],
    ) -> tuple[Coordinate, ...]:
        candidates = sorted(
            coordinates[self.layout.anchor_level],
            key=lambda point: (point[1], point[0]),
        )
        random.Random(f"{self.selection_seed}:{slide_name}").shuffle(candidates)
        complete: list[Coordinate] = []
        accepted: list[Coordinate] = []
        missing_by_level: dict[int, int] = {}
        for anchor in candidates:
            for level, required in self.layout.coordinates_by_level(anchor).items():
                if not coordinates[level].issuperset(required):
                    missing_by_level[level] = missing_by_level.get(level, 0) + 1
                    break
            else:
                complete.append(anchor)
                if self.overlap_policy is OverlapPolicy.ALL or not any(
                    self._overlaps(anchor, retained) for retained in accepted
                ):
                    accepted.append(anchor)

        self.diagnostics[slide_name] = PyramidDiagnostics(
            candidates=len(candidates),
            complete=len(complete),
            missing_by_level=missing_by_level,
        )
        return tuple(accepted)

    def _overlaps(self, first: Coordinate, second: Coordinate) -> bool:
        """Return whether L5 footprints intersect with strictly positive area."""
        first_x, first_y = first
        second_x, second_y = second
        extent = self.layout.tile_extent
        return (
            first_x < second_x + extent
            and second_x < first_x + extent
            and first_y < second_y + extent
            and second_y < first_y + extent
        )

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> PyramidSample:
        row = self.slides.iloc[idx]
        slide_name = str(row["name"])
        embeddings_by_level: dict[int, dict[Coordinate, Tensor]] = {}
        feature_dim: int | None = None

        for level in self.layout.level_order:
            frame = pd.read_parquet(self._parquet_path(level, slide_name))
            required_columns = {"x", "y", "embedding"}
            missing_columns = required_columns.difference(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"Slide {slide_name!r} level {level} is missing columns "
                    f"{sorted(missing_columns)}."
                )

            level_embeddings: dict[Coordinate, Tensor] = {}
            for x, y, embedding in frame[["x", "y", "embedding"]].itertuples(
                index=False, name=None
            ):
                coordinate = (int(x), int(y))
                if coordinate in level_embeddings:
                    raise ValueError(
                        f"Slide {slide_name!r} level {level} contains a duplicate "
                        f"coordinate {coordinate}."
                    )
                vector = torch.from_numpy(np.array(embedding, copy=True)).float()
                if vector.ndim != 1:
                    raise ValueError(
                        f"Slide {slide_name!r} level {level} has a non-vector "
                        f"embedding at {coordinate}."
                    )
                if feature_dim is None:
                    feature_dim = vector.numel()
                elif vector.numel() != feature_dim:
                    raise ValueError(
                        f"Slide {slide_name!r} level {level} embedding width "
                        f"{vector.numel()} does not match {feature_dim}."
                    )
                level_embeddings[coordinate] = vector
            embeddings_by_level[level] = level_embeddings

        regions: list[Tensor] = []
        for anchor in self.anchors_by_slide[slide_name]:
            tokens: list[Tensor] = []
            for level, coordinates in self.layout.coordinates_by_level(anchor).items():
                for coordinate in coordinates:
                    try:
                        tokens.append(embeddings_by_level[level][coordinate])
                    except KeyError as error:
                        raise RuntimeError(
                            f"Slide {slide_name!r} level {level} lost coordinate "
                            f"{coordinate} after its completeness scan."
                        ) from error
            regions.append(torch.stack(tokens))

        bag = PyramidBag(torch.stack(regions).float())
        label = get_label(row, self.label_mode)
        metadata: PyramidSlideMetadata = {
            "slide_id": slide_name,
            "anchors": self.anchors_by_slide[slide_name],
        }
        return bag, label, metadata


__all__ = ["PyramidDiagnostics", "PyramidLayout", "PyramidScaleDataset"]
