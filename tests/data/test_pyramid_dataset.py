from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.data.datasets.pyramid import PyramidScaleDataset


if TYPE_CHECKING:
    from ml.typing import LevelSpec


def _code(level: int, x: int, y: int) -> float:
    return float(level * 10_000 + y * 100 + x)


class PyramidScaleDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dirs = {level: self.root / f"l{level}" for level in (5, 4, 3)}
        for directory in self.dirs.values():
            directory.mkdir()
        self.mapping = self.root / "mapping.csv"
        pd.DataFrame(
            {
                "path": ["/slides/slide_a.svs"],
                "type": ["a luminal"],
                "mammaprint_index": [0.5],
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
        *,
        anchors: tuple[tuple[int, int], ...] = ((0, 0),),
        missing: tuple[int, int, int] | None = None,
        duplicate: tuple[int, int, int] | None = None,
        width_by_level: dict[int, int] | None = None,
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
        widths = width_by_level or {}
        for level, points in coordinates.items():
            output_points = tuple(points)
            if duplicate is not None and level == duplicate[0]:
                output_points += ((duplicate[1], duplicate[2]),)
            width = widths.get(level, 2)
            frame = pd.DataFrame(
                {
                    "x": [x for x, _ in output_points],
                    "y": [y for _, y in output_points],
                    "embedding": [
                        np.full(width, _code(level, x, y), dtype=np.float32)
                        for x, y in output_points
                    ],
                }
            )
            frame.to_parquet(self.dirs[level] / "slide_a.parquet", index=False)

    def _dataset(
        self,
        *,
        require_complete: bool = True,
        overlap_policy: str = "non_overlapping",
        selection_seed: int = 0,
    ) -> PyramidScaleDataset:
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
                level_ratios=(1, 2, 4),
                label_mode="type",
                require_complete=require_complete,
                overlap_policy=overlap_policy,
                selection_seed=selection_seed,
            )

    def test_returns_complete_pyramid_in_level_then_row_major_order(self) -> None:
        self._write_slide()

        dataset = self._dataset()
        bag, label, metadata = dataset[0]

        self.assertEqual(tuple(bag.shape), (1, 21, 2))
        self.assertEqual(label.tolist(), [1.0])
        self.assertEqual(metadata, {"slide_id": "slide_a", "anchors": ((0, 0),)})
        self.assertEqual(
            bag[0, :, 0].tolist(),
            [
                50000.0,
                40000.0,
                40002.0,
                40200.0,
                40202.0,
                30000.0,
                30002.0,
                30004.0,
                30006.0,
                30200.0,
                30202.0,
                30204.0,
                30206.0,
                30400.0,
                30402.0,
                30404.0,
                30406.0,
                30600.0,
                30602.0,
                30604.0,
                30606.0,
            ],
        )
        self.assertEqual(dataset.diagnostics["slide_a"].complete, 1)

    def test_removes_slide_when_fine_tile_is_missing(self) -> None:
        self._write_slide(missing=(3, 6, 6))

        dataset = self._dataset()

        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.diagnostics["slide_a"].missing_by_level, {3: 1})

    def test_rejects_duplicate_coordinates_during_scan(self) -> None:
        self._write_slide(duplicate=(4, 0, 0))

        with self.assertRaisesRegex(ValueError, "duplicate coordinate"):
            self._dataset()

    def test_rejects_inconsistent_embedding_widths_when_gathering(self) -> None:
        self._write_slide(width_by_level={3: 3})
        dataset = self._dataset()

        with self.assertRaisesRegex(ValueError, "embedding width"):
            dataset[0]

    def test_metadata_anchors_match_the_returned_bag_region_order(self) -> None:
        self._write_slide(anchors=((0, 0), (2, 0)))

        dataset = self._dataset(overlap_policy="all")
        bag, _, metadata = dataset[0]

        self.assertEqual(len(metadata["anchors"]), 2)
        self.assertEqual(
            [bag[index, 0, 0].item() for index in range(2)],
            [50_000.0 + anchor[0] for anchor in metadata["anchors"]],
        )

    def test_non_overlapping_selection_removes_intersecting_l5_footprints(self) -> None:
        self._write_slide(anchors=((0, 0), (1, 0)))

        dataset = self._dataset()

        self.assertEqual(len(dataset.anchors_by_slide["slide_a"]), 1)

    def test_non_overlapping_selection_keeps_boundary_touching_footprints(self) -> None:
        self._write_slide(anchors=((0, 0), (2, 0)))

        dataset = self._dataset()

        self.assertEqual(set(dataset.anchors_by_slide["slide_a"]), {(0, 0), (2, 0)})

    def test_all_overlap_policy_retains_intersecting_footprints(self) -> None:
        self._write_slide(anchors=((0, 0), (1, 0)))

        dataset = self._dataset(overlap_policy="all")

        self.assertEqual(set(dataset.anchors_by_slide["slide_a"]), {(0, 0), (1, 0)})

    def test_selection_seed_reproduces_anchor_order(self) -> None:
        self._write_slide(anchors=((0, 0), (1, 0), (2, 0), (3, 0)))

        first = self._dataset(selection_seed=91)
        second = self._dataset(selection_seed=91)

        self.assertEqual(
            first.anchors_by_slide["slide_a"], second.anchors_by_slide["slide_a"]
        )

    def test_rejects_incomplete_pyramid_mode(self) -> None:
        self._write_slide()

        with self.assertRaisesRegex(ValueError, "require_complete"):
            self._dataset(require_complete=False)


if __name__ == "__main__":
    unittest.main()
