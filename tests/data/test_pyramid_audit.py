from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.data.datasets.pyramid import PyramidScaleDataset


if TYPE_CHECKING:
    from ml.typing import LevelSpec


class PyramidAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dirs = {level: self.root / f"l{level}" for level in (5, 4, 3)}
        for directory in self.dirs.values():
            directory.mkdir()
        self.mapping = self.root / "mapping.csv"
        pd.DataFrame(
            {
                "path": [
                    "/slides/complete.svs",
                    "/slides/incomplete.svs",
                    "/slides/overlap.svs",
                    "/slides/zero.svs",
                ],
                "type": ["a luminal"] * 4,
                "mammaprint_index": [0.5] * 4,
            }
        ).to_csv(self.mapping, index=False)
        self.levels: dict[int, LevelSpec] = {
            level: {
                "mpp": {5: 8.0, 4: 4.0, 3: 2.0}[level],
                "tile_extent": 2,
                "uris": {"train": f"unused://l{level}"},
            }
            for level in (5, 4, 3)
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_slide(
        self,
        slide_id: str,
        anchors: tuple[tuple[int, int], ...],
        *,
        missing: tuple[int, int, int] | None = None,
    ) -> None:
        coordinates: dict[int, set[tuple[int, int]]] = {
            5: set(anchors),
            4: set(),
            3: set(),
        }
        for x, y in anchors:
            coordinates[4].update(
                (2 * x + column * 2, 2 * y + row * 2)
                for row in range(2)
                for column in range(2)
            )
            coordinates[3].update(
                (4 * x + column * 2, 4 * y + row * 2)
                for row in range(4)
                for column in range(4)
            )
        if missing is not None:
            level, x, y = missing
            coordinates[level].remove((x, y))
        for level, points in coordinates.items():
            ordered_points = sorted(points)
            pd.DataFrame(
                {
                    "x": [x for x, _ in ordered_points],
                    "y": [y for _, y in ordered_points],
                    "embedding": [
                        np.full(2, level * 10_000 + y * 100 + x, dtype=np.float32)
                        for x, y in ordered_points
                    ],
                }
            ).to_parquet(self.dirs[level] / f"{slide_id}.parquet", index=False)

    def _dataset(self) -> PyramidScaleDataset:
        with patch(
            "ml.data.datasets.pyramid.download_level_sources",
            return_value=self.dirs,
        ):
            return PyramidScaleDataset(
                split="train",
                levels=self.levels,
                data_mapping=self.mapping,
                anchor_level=5,
                level_order=(5, 4, 3),
                tile_extent=2,
                label_mode="type",
                overlap_policy="non_overlapping",
                selection_seed=0,
            )

    def test_audit_records_report_complete_incomplete_overlap_and_zero_retention(
        self,
    ) -> None:
        self._write_slide("complete", ((0, 0),))
        self._write_slide("incomplete", ((0, 0),), missing=(3, 6, 6))
        self._write_slide("overlap", ((0, 0), (1, 0)))
        self._write_slide("zero", ())

        dataset = self._dataset()
        records = {record.slide_id: record for record in dataset.audit_records}

        self.assertEqual(
            asdict(records["complete"]),
            {
                "slide_id": "complete",
                "split": "train",
                "label": "a luminal",
                "candidates": 1,
                "complete_before_thinning": 1,
                "retained": 1,
                "incomplete": 0,
                "missing_by_level": {},
                "completion_rate": 1.0,
                "overlap_retention_rate": 1.0,
                "rejected": False,
            },
        )
        self.assertEqual(
            asdict(records["incomplete"]),
            {
                "slide_id": "incomplete",
                "split": "train",
                "label": "a luminal",
                "candidates": 1,
                "complete_before_thinning": 0,
                "retained": 0,
                "incomplete": 1,
                "missing_by_level": {3: 1},
                "completion_rate": 0.0,
                "overlap_retention_rate": 0.0,
                "rejected": True,
            },
        )
        self.assertEqual(
            asdict(records["overlap"]),
            {
                "slide_id": "overlap",
                "split": "train",
                "label": "a luminal",
                "candidates": 2,
                "complete_before_thinning": 2,
                "retained": 1,
                "incomplete": 0,
                "missing_by_level": {},
                "completion_rate": 1.0,
                "overlap_retention_rate": 0.5,
                "rejected": False,
            },
        )
        self.assertEqual(
            asdict(records["zero"]),
            {
                "slide_id": "zero",
                "split": "train",
                "label": "a luminal",
                "candidates": 0,
                "complete_before_thinning": 0,
                "retained": 0,
                "incomplete": 0,
                "missing_by_level": {},
                "completion_rate": 0.0,
                "overlap_retention_rate": 0.0,
                "rejected": True,
            },
        )

    def test_audit_tables_aggregate_totals_rates_and_retained_quantiles(self) -> None:
        self._write_slide("complete", ((0, 0),))
        self._write_slide("incomplete", ((0, 0),), missing=(3, 6, 6))
        self._write_slide("overlap", ((0, 0), (1, 0)))
        self._write_slide("zero", ())

        dataset = self._dataset()
        frame = dataset.audit_frame()
        summary = dataset.audit_summary().iloc[0]

        self.assertEqual(frame.shape, (4, 11))
        self.assertEqual(summary["split"], "train")
        self.assertEqual(summary["label"], "a luminal")
        self.assertEqual(summary["slides"], 4)
        self.assertEqual(summary["rejected_slides"], 2)
        self.assertEqual(summary["candidates"], 4)
        self.assertEqual(summary["complete_before_thinning"], 3)
        self.assertEqual(summary["retained"], 2)
        self.assertEqual(summary["incomplete"], 1)
        self.assertEqual(summary["completion_rate"], 0.75)
        self.assertEqual(summary["overlap_retention_rate"], 2 / 3)
        self.assertEqual(summary["retained_p25"], 0.0)
        self.assertEqual(summary["retained_p50"], 0.5)
        self.assertEqual(summary["retained_p75"], 1.0)


if __name__ == "__main__":
    unittest.main()
