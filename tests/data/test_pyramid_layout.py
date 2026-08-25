from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from ml.data.datasets.pyramid import PyramidLayout


if TYPE_CHECKING:
    from ml.typing import LevelSpec


class PyramidLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.levels: dict[int, LevelSpec] = {
            5: {"mpp": 8.0, "tile_extent": 256, "uris": {}},
            4: {"mpp": 4.0, "tile_extent": 256, "uris": {}},
            3: {"mpp": 2.0, "tile_extent": 256, "uris": {}},
        }

    def test_grid_anchor_three_maps_to_one_four_sixteen_tiles(self) -> None:
        layout = PyramidLayout.from_levels(
            self.levels,
            anchor_level=5,
            level_order=(5, 4, 3),
            tile_extent=256,
        )

        groups = layout.coordinates_by_level((3 * 256, 3 * 256))

        self.assertEqual(groups[5], ((768, 768),))
        self.assertEqual(
            groups[4],
            (
                (1536, 1536),
                (1792, 1536),
                (1536, 1792),
                (1792, 1792),
            ),
        )
        self.assertEqual(
            groups[3],
            (
                (3072, 3072),
                (3328, 3072),
                (3584, 3072),
                (3840, 3072),
                (3072, 3328),
                (3328, 3328),
                (3584, 3328),
                (3840, 3328),
                (3072, 3584),
                (3328, 3584),
                (3584, 3584),
                (3840, 3584),
                (3072, 3840),
                (3328, 3840),
                (3584, 3840),
                (3840, 3840),
            ),
        )
        self.assertEqual(len(layout.coordinates((768, 768))), 21)

    def test_rejects_non_integer_ratio(self) -> None:
        self.levels[4]["mpp"] = 3.0

        with self.assertRaisesRegex(ValueError, "integer scale ratio"):
            PyramidLayout.from_levels(
                self.levels,
                anchor_level=5,
                level_order=(5, 4, 3),
                tile_extent=256,
            )

    def test_rejects_level_tile_extent_mismatch(self) -> None:
        self.levels[3]["tile_extent"] = 224

        with self.assertRaisesRegex(ValueError, "tile_extent"):
            PyramidLayout.from_levels(
                self.levels,
                anchor_level=5,
                level_order=(5, 4, 3),
                tile_extent=256,
            )

    def test_requires_anchor_to_be_first_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "start with anchor_level"):
            PyramidLayout.from_levels(
                self.levels,
                anchor_level=5,
                level_order=(4, 5, 3),
                tile_extent=256,
            )

    def test_normalizes_hydra_level_sequence_to_tuple(self) -> None:
        layout = PyramidLayout.from_levels(
            self.levels,
            anchor_level=5,
            level_order=[5, 4, 3],
            tile_extent=256,
        )

        self.assertIsInstance(layout.level_order, tuple)
        self.assertEqual(layout.level_order, (5, 4, 3))


if __name__ == "__main__":
    unittest.main()
