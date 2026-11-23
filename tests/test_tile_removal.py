import unittest
from itertools import combinations

import numpy as np

from ml.explainability.tile_removal import class_removal_order, minimum_class_mask
from ml.explainability.tile_removal_index import minimum_index_mask
from ml.explainability.tile_removal_regions import (
    TileGrid,
    compact_class_mask,
    matched_controls,
    union_area,
)


def reference_index(scores, logits, keep=None):
    scores, logits = np.asarray(scores), np.asarray(logits)
    if keep is not None:
        scores, logits = scores[keep], logits[keep]
    return np.average(scores, weights=np.exp(logits - logits.max()))


def oracle(scores, logits, predicate, budget, protected=()):
    for k in range(min(budget, len(scores) - 1) + 1):
        deletable = [i for i in range(len(scores)) if i not in protected]
        for removed in combinations(deletable, k):
            keep = np.ones(len(scores), dtype=bool)
            keep[list(removed)] = False
            if predicate(reference_index(scores, logits, keep)):
                return k
    return None


class TileRemovalTest(unittest.TestCase):
    def test_prefix_minimality_against_exhaustive_masks(self):
        rng = np.random.default_rng(42)
        for weighted in (False, True):
            for _ in range(50):
                length = int(rng.integers(1, 8))
                scores = rng.normal(size=length) + 0.1
                logits = rng.normal(size=length) * int(weighted)
                tile_ids = rng.permutation(length)
                protected_rows = (0,) if length > 2 else ()
                protected = tuple(int(tile_ids[i]) for i in protected_rows)
                budget = int(rng.integers(length))
                for target in ("A", "B"):
                    predicate = (
                        (lambda x: x >= 0.03) if target == "A" else (lambda x: x < 0.03)
                    )
                    expected = oracle(scores, logits, predicate, budget, protected_rows)
                    result = minimum_class_mask(
                        scores,
                        logits,
                        tile_ids,
                        target,
                        threshold=0.03,
                        max_removed=budget,
                        protected_ids=protected,
                    )
                    if expected is None:
                        self.assertEqual(result.status, "infeasible")
                    else:
                        self.assertEqual(result.status, "minimum")
                        self.assertEqual(result.count, expected)
                        self.assertTrue(predicate(result.index))
                        self.assertTrue(set(result.removed_ids).isdisjoint(protected))

    def test_weighted_contribution_ranks_over_raw_score(self):
        result = minimum_class_mask(
            [-0.60, -0.20, 0.10], np.log([0.01, 0.50, 0.49]), [10, 11, 12], "A"
        )
        self.assertEqual(result.removed_ids, (11,))
        self.assertGreater(result.index, 0)

    def test_zero_is_a_and_ties_are_ordered_by_tile_id(self):
        scores, logits, ids = [-1.0, -1.0, 1.0], [0, 0, 0], [9, 3, 5]
        kwargs = {"threshold": 0.0}
        result = minimum_class_mask(scores, logits, ids, "A", **kwargs)
        self.assertEqual(
            (result.removed_ids, result.index, result.status), ((3,), 0, "minimum")
        )
        self.assertEqual(minimum_class_mask(scores, logits, ids, "A", **kwargs), result)
        reverse = minimum_class_mask([-1, 1], [0, 0], [4, 8], "B")
        self.assertEqual((reverse.removed_ids, reverse.index), ((8,), -1))

    def test_strict_boundary_above_budget_and_single_tile(self):
        scores = [1] * 4 + [-0.1]
        result = minimum_class_mask(scores, [0] * 5, np.arange(5), "B")
        self.assertEqual(result.count, 4)
        self.assertEqual(
            minimum_class_mask(
                scores, [0] * 5, np.arange(5), "B", max_removed=0
            ).status,
            "infeasible",
        )
        self.assertEqual(minimum_class_mask([0], [0], [2], "B").status, "infeasible")
        self.assertEqual(
            minimum_class_mask([-1, -0.1], [0, 0], [2, 3], "A").status, "infeasible"
        )

    def test_log_mass_renormalizes_after_extreme_deletions(self):
        result = minimum_class_mask([-1, -0.5, 1], [2000, 1000, 0], [4, 5, 6], "A")
        self.assertEqual((result.removed_ids, result.index), ((4, 5), 1.0))
        cancellation = minimum_class_mask([1, -1, -1], [0, 0, -1000], [1, 2, 3], "B")
        self.assertEqual(
            (cancellation.removed_ids, cancellation.status), ((), "minimum")
        )

    def test_native_fallback_order_includes_neutral_and_favored_tiles(self):
        scores, logits, ids = [5, -4, 0, 1, 1, 8], [0] * 6, [1, 2, 3, 7, 5, 4]
        self.assertEqual(
            class_removal_order(scores, logits, ids, "A"), (2, 3, 5, 7, 1, 4)
        )
        order = class_removal_order(scores, logits, ids, "B", protected_ids=(4, 3))
        self.assertEqual(order, (1, 5, 7, 2))
        first = minimum_class_mask(scores, logits, ids, "B")
        self.assertEqual(
            class_removal_order(scores, logits, ids, "B")[: first.count],
            first.removed_ids,
        )


class TileRemovalRegionsTest(unittest.TestCase):
    def test_overlapping_footprints_and_hole(self):
        grid = TileGrid([[0, 0, 2, 2], [1, 0, 3, 2], [3, 0, 5, 2]], 1, (2,))
        self.assertEqual(union_area(grid.rectangles), 10)
        self.assertEqual(union_area(grid.rectangles[:2]), 6)
        result = compact_class_mask(
            [-3, -3, 1], [0, 0, 0], [8, 3, 9], "A", grid, max_removed=2
        )
        self.assertEqual(
            (result.removed_ids, result.window, result.status),
            ((3, 8), (0, 0, 2), "minimum"),
        )
        fallback = compact_class_mask(
            [-3, -3, 1], [0, 0, 0], [8, 3, 9], "A", grid, max_removed=1
        )
        self.assertEqual(
            (fallback.removed_ids, fallback.index, fallback.status),
            ((3,), -1, "infeasible"),
        )

    def test_all_windows_agree_with_grid_oracle(self):
        rng = np.random.default_rng(20)
        for _ in range(30):
            cells = sorted(rng.choice(30, 18, replace=False))
            rectangles = [[i % 6, i // 6, i % 6 + 2, i // 6 + 2] for i in cells]
            grid = TileGrid(rectangles, 1, (2, 3, 5))
            scores = rng.normal(size=len(cells))
            logits = rng.normal(size=len(cells)) * 10
            ids = np.arange(len(cells))
            for target, direction in (("A", 1), ("B", -1)):
                predicate = (lambda x: x >= 0) if target == "A" else (lambda x: x < 0)
                candidates = [((), reference_index(scores, logits))]
                for side in grid.sizes:
                    for y in range(grid.shape[0]):
                        for x in range(grid.shape[1]):
                            rows = grid.rows(x, y, side)
                            if 0 < len(rows) <= 8:
                                kept = ~np.isin(ids, rows)
                                candidates.append(
                                    (
                                        tuple(rows),
                                        reference_index(scores, logits, kept),
                                    )
                                )
                result = compact_class_mask(
                    scores, logits, ids, target, grid, max_removed=8
                )
                successful = [rows for rows, value in candidates if predicate(value)]
                if successful:
                    minimum = min(map(len, successful))
                    self.assertEqual(
                        (result.status, result.count), ("minimum", minimum)
                    )
                else:
                    best = max(value * direction for _, value in candidates[1:])
                    self.assertEqual(result.status, "infeasible")
                    self.assertAlmostEqual(result.index * direction, best)

    def test_control_selection_is_seeded_and_matches_masks(self):
        grid = TileGrid([[x, 0, x + 2, 2] for x in range(6)], 1, (2,))
        ids = np.arange(6)

        def epithelium(rows):
            return (sum(i < 4 for i in rows) / len(rows),)

        kwargs = {
            "compact": True,
            "seed": 3,
            "window_size": 2,
            "repeats": 20,
            "coverage": epithelium,
            "coverage_tolerance": 0.0,
            "area_tolerance": 0.0,
        }
        controls = matched_controls(ids, grid, (0, 1), **kwargs)
        self.assertEqual(set(controls), {(1, 2), (2, 3)})
        self.assertEqual(matched_controls(ids, grid, (0, 1), **kwargs), controls)

    def test_unsuitable_grid_reports_geometry_error(self):
        with self.assertRaisesRegex(ValueError, "source grid"):
            TileGrid([[0, 0, 4, 4], [3, 0, 7, 4]], 2)

    def test_explicit_native_mismatch_still_scans_nonempty_windows(self):
        grid = TileGrid([[x, 0, x + 1, 1] for x in range(3)], 1, (2,))
        input_values = ([0, 0, 0], [0, 0, 0], [5, 8, 3], "A", grid)
        self.assertEqual(
            compact_class_mask(*input_values, max_removed=1).removed_ids, ()
        )
        required = compact_class_mask(
            *input_values, require_removal=True, max_removed=1
        )
        self.assertEqual((required.removed_ids, required.status), ((3,), "minimum"))

    def test_control_area_can_use_verified_level0_footprints(self):
        grid = TileGrid([[x, 0, x + 1, 1] for x in range(3)], 1, (2,))
        controls = matched_controls(
            [0, 1, 2],
            grid,
            (0,),
            compact=False,
            seed=5,
            area_tolerance=0.0,
            area_rectangles=[[0, 0, 2, 2], [2, 0, 4, 2], [4, 0, 5, 2]],
        )
        self.assertEqual(controls, ((1,),))


class TileRemovalIndexTest(unittest.TestCase):
    def test_class_prefix_overshoots_but_two_deletions_hit_band(self):
        scores = [-2.0, 0.0, 0.3, 1.5]
        logits = [0] * 4
        tile_ids = [10, 11, 12, 13]
        crossing = minimum_class_mask(scores, logits, tile_ids, "A", threshold=0.1)
        self.assertGreater(crossing.index, 0.2)
        result = minimum_index_mask(scores, logits, tile_ids, 0.1, 0.2, max_removed=2)
        self.assertEqual(
            (result.removed_ids, result.index, result.status),
            ((10, 13), 0.15, "minimum"),
        )

    def test_inclusive_and_already_satisfied_band(self):
        exact = minimum_index_mask([-1, 1], [0, 0], [6, 8], 1, 1, max_removed=1)
        self.assertEqual(
            (exact.removed_ids, exact.index, exact.status), ((6,), 1, "minimum")
        )
        self.assertEqual(
            minimum_index_mask([-1, 1], [0, 0], [6, 8], 0, 0, max_removed=1).count, 0
        )

    def test_exhaustive_oracle_under_weighted_deletions(self):
        rng = np.random.default_rng(4)
        for _ in range(35):
            n = int(rng.integers(1, 8))
            scores = rng.normal(size=n)
            logits = rng.normal(size=n)
            budget = int(rng.integers(n))
            lower, upper = sorted(rng.normal(size=2))
            expected = oracle(
                scores,
                logits,
                lambda x, low=lower, high=upper: low <= x <= high,
                budget,
            )
            result = minimum_index_mask(
                scores, logits, np.arange(n), lower, upper, max_removed=budget
            )
            if expected is None:
                self.assertEqual(result.status, "infeasible")
                self.assertIsNotNone(result.distance_bounds)
                low, high = result.distance_bounds
                self.assertEqual(low, high)
            else:
                self.assertEqual((result.count, result.status), (expected, "minimum"))

    def test_integer_solver_and_closest_result(self):
        ids = np.arange(13)
        result = minimum_index_mask([5] + [0] * 12, [0] * 13, ids, 0, 0, max_removed=2)
        self.assertEqual((result.removed_ids, result.status), ((0,), "minimum"))
        unreachable = minimum_index_mask(
            [2] + [1] * 12, [0] * 13, ids, 3, 4, max_removed=1
        )
        self.assertEqual(unreachable.status, "infeasible")
        self.assertAlmostEqual(unreachable.index, 13 / 12)
        low, high = unreachable.distance_bounds
        self.assertLessEqual(low, 3 - 13 / 12)
        self.assertGreaterEqual(high, 3 - 13 / 12)
        self.assertLessEqual(high - low, 1e-4)

    def test_timeout_keeps_a_usable_candidate(self):
        result = minimum_index_mask(
            [-2.0, 0.0, 0.3, 1.5],
            [0] * 4,
            np.arange(4),
            0.1,
            0.2,
            max_removed=2,
            time_limit=1e-12,
        )
        self.assertEqual(result.status, "unresolved")
        self.assertAlmostEqual(result.index, -0.05)


if __name__ == "__main__":
    unittest.main()
