"""Compact deletion windows and same-slide controls on the source tile grid."""

from collections.abc import Callable
from itertools import pairwise
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml.explainability.tile_removal import (
    FloatArray,
    IntArray,
    MaskResult,
    _budget,
    _margin_sign,
    _tile_ids,
    _values,
    attention_weights,
    pooled_index,
)


_MAX_GRID_CELLS = 4_000_000
Coverage = Callable[[tuple[int, ...]], tuple[float, ...]]


def _rectangles(rectangles: ArrayLike) -> IntArray:
    value = np.asarray(rectangles)
    if value.shape == (0,):
        return value.reshape(0, 4).astype(np.int64)
    if (
        value.ndim != 2
        or value.shape[1] != 4
        or value.dtype.kind not in "iu"
        or (value < 0).any()
        or (value > np.iinfo(np.int64).max).any()
        or (value[:, 2:] <= value[:, :2]).any()
    ):
        raise ValueError(
            "Expected integer (x0, y0, x1, y1) rectangles of positive area."
        )
    return value.astype(np.int64, copy=False)


def union_area(rectangles: ArrayLike) -> int:
    """Return exact area of the union of half-open rectangles."""
    rectangles = _rectangles(rectangles)
    if not len(rectangles):
        return 0
    y, inverse = np.unique(rectangles[:, [1, 3]], return_inverse=True)
    y_segments = np.diff(y)
    inverse = inverse.reshape(-1, 2)
    event_x = np.r_[rectangles[:, 0], rectangles[:, 2]]
    event_rows = np.argsort(event_x, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(event_x[event_rows])) + 1, len(event_rows)]
    deltas = np.zeros(len(y), dtype=np.int64)
    area = 0
    last_x = int(event_x[event_rows[0]])
    count = len(rectangles)
    for start, end in pairwise(cuts):
        events = event_rows[start:end]
        x = int(event_x[events[0]])
        if x != last_x:
            height = int(y_segments[deltas.cumsum()[:-1] > 0].sum())
            area += (x - last_x) * height
        rows = events % count
        direction = np.where(events < count, 1, -1)
        np.add.at(deltas, inverse[rows, 0], direction)
        np.add.at(deltas, inverse[rows, 1], -direction)
        last_x = x
    return area


def _prefix[DType: (np.int64, np.float64)](grid: NDArray[DType]) -> NDArray[DType]:
    result = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=grid.dtype)
    result[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
    return result


def _windows[DType: (np.int64, np.float64)](
    prefix: NDArray[DType], size: int
) -> NDArray[DType]:
    height, width = prefix.shape[0] - 1, prefix.shape[1] - 1
    x_end = np.minimum(np.arange(width) + size, width)
    y_end = np.minimum(np.arange(height) + size, height)
    return (
        prefix[y_end[:, None], x_end[None, :]]
        - prefix[y_end, :-1]
        - prefix[:-1, x_end]
        + prefix[:-1, :-1]
    )


class TileGrid:
    """One regular level-0 lattice, retaining source row order and edge clipping."""

    def __init__(
        self,
        rectangles: ArrayLike,
        stride: int | tuple[int, int],
        sizes: tuple[int, ...] = (2, 4, 8, 16, 32),
    ) -> None:
        rectangles = _rectangles(rectangles)
        if not len(rectangles):
            raise ValueError("A tile grid must have at least one rectangle.")
        if isinstance(stride, Integral):
            stride_pair = (int(stride), int(stride))
        elif isinstance(stride, tuple):
            stride_pair = stride
        else:
            raise ValueError("Grid stride must be an integer or (x, y) pair.")
        if (
            len(stride_pair) != 2
            or any(not isinstance(s, Integral) or s < 1 for s in stride_pair)
            or not sizes
            or any(not isinstance(s, Integral) or s < 1 for s in sizes)
        ):
            raise ValueError("Grid stride and square sizes must be positive integers.")
        origins = rectangles[:, :2]
        offsets = origins - origins.min(axis=0)
        stride_array = np.asarray(stride_pair, dtype=np.int64)
        if (offsets % stride_array).any():
            raise ValueError("Tile origins do not lie on the source grid.")
        positions = offsets // stride_array
        x_max, y_max = map(int, positions.max(axis=0))
        self.shape = (y_max + 1, x_max + 1)
        if self.shape[0] * self.shape[1] > _MAX_GRID_CELLS:
            raise ValueError("Tile grid exceeds the supported number of cells.")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("Duplicate tile origins make spatial windows ambiguous.")
        self.rectangles = rectangles
        self.positions = positions
        self.sizes = tuple(sorted(set(sizes)))
        self._rows = np.full(self.shape, -1, dtype=np.int64)
        self._rows[positions[:, 1], positions[:, 0]] = np.arange(len(positions))
        self._count_prefix = _prefix((self._rows >= 0).astype(np.int64))

    def counts(self, size: int) -> IntArray:
        return _windows(self._count_prefix, size)

    def rows(self, x: int, y: int, size: int) -> IntArray:
        rows = self._rows[y : y + size, x : x + size].ravel()
        return np.sort(rows[rows >= 0])

    def prefix(self, values: FloatArray) -> FloatArray:
        field = np.zeros(self.shape, dtype=np.float64)
        field[self.positions[:, 1], self.positions[:, 0]] = values
        return _prefix(field)


def compact_class_mask(
    scores: ArrayLike,
    log_weights: ArrayLike,
    tile_ids: ArrayLike,
    target_class: str,
    grid: TileGrid,
    *,
    threshold: float = 0.0,
    max_removed: int,
    require_removal: bool = False,
) -> MaskResult:
    """Scan all declared squares and return the smallest feasible window.

    Set ``require_removal`` when the caller has verified that the native
    intact prediction misses the target despite a mathematical boundary tie.
    """
    scores, log_weights = _values(scores, log_weights)
    ids = _tile_ids(tile_ids, len(scores))
    budget = _budget(max_removed, len(scores))
    if len(grid.rectangles) != len(scores):
        raise ValueError("Tile grid must preserve score row order and length.")
    if target_class not in ("A", "B") or not np.isfinite(threshold):
        raise ValueError("Expected target A or B and a finite cutoff.")
    intact = pooled_index(scores, log_weights)
    initial_sign = _margin_sign(scores, log_weights, threshold)
    initially_reached = (target_class == "A" and initial_sign in (0, 1)) or (
        target_class == "B" and initial_sign == -1
    )
    if initially_reached and not require_removal:
        return MaskResult((), intact, "minimum")
    if budget == 0:
        return MaskResult(
            (), intact, "unresolved" if initial_sign is None else "infeasible"
        )

    direction = 1 if target_class == "A" else -1
    # Prefix sums screen the windows in bulk. Recompute ambiguous signs on
    # the retained rows so subtraction and underflow cannot certify a mask.
    scale = max(float(np.max(np.abs(scores))), abs(threshold), np.finfo(float).tiny)
    weights = attention_weights(log_weights)
    values = (scores / scale) * weights
    weights_prefix = grid.prefix(weights)
    value_prefix = grid.prefix(values)
    total_weight = float(weights_prefix[-1, -1])
    total_value = float(value_prefix[-1, -1])
    cutoff = threshold / scale
    error = np.finfo(float).eps * 32 * (sum(grid.shape) + 1)
    best: tuple[tuple[int, ...], MaskResult] | None = None
    closest: tuple[tuple[float, ...], MaskResult] | None = None
    unknown_at = 0 if initial_sign is None else len(scores)
    constant = bool(np.all(scores == scores[0]))

    for size in grid.sizes:
        counts = grid.counts(size)
        eligible = (counts > 0) & (counts <= budget)
        if not eligible.any():
            continue
        remaining_weight = total_weight - _windows(weights_prefix, size)
        remaining_value = total_value - _windows(value_prefix, size)
        margin = direction * (remaining_value - cutoff * remaining_weight)
        reliable = (np.abs(margin) > error) & (remaining_weight > error)
        sure = eligible & reliable & (margin > 0)
        optimistic = eligible & (sure | ~reliable)
        if sure.any():
            optimistic &= counts <= counts[sure].min()
        if constant and not initially_reached:
            # The intact bag already established that a constant cannot flip.
            optimistic.fill(False)
        if best is not None:
            optimistic &= counts <= best[0][0]
        ys, xs = np.nonzero(optimistic)
        for i in np.argsort(counts[ys, xs], kind="stable"):
            y, x = int(ys[i]), int(xs[i])
            count = int(counts[y, x])
            if best is not None and count > best[0][0]:
                break
            rows = grid.rows(x, y, size)
            sign: int | None = direction
            if constant:
                sign = initial_sign
            elif not reliable[y, x]:
                keep = np.ones(len(scores), dtype=bool)
                keep[rows] = False
                sign = _margin_sign(scores[keep], log_weights[keep], threshold)
            if sign is None:
                unknown_at = min(unknown_at, count)
                continue
            if direction * sign > 0 or (sign == 0 and target_class == "A"):
                key = (count, union_area(grid.rectangles[rows]), size, y, x)
                if best is None or key < best[0]:
                    keep = np.ones(len(scores), dtype=bool)
                    keep[rows] = False
                    best = (
                        key,
                        MaskResult(
                            tuple(sorted(map(int, ids[rows]))),
                            pooled_index(scores, log_weights, keep),
                            "minimum",
                            window=(x, y, size),
                        ),
                    )

        if best is not None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            approximation = direction * remaining_value / remaining_weight
        stable = eligible & (remaining_weight > error)
        uncertain = eligible & ~stable
        if stable.any():
            # Include every candidate whose numerical interval overlaps the
            # best estimate. The retained-bag computation breaks close ties.
            score_error = np.zeros_like(remaining_weight)
            np.divide(2 * error, remaining_weight, out=score_error, where=stable)
            best_lower = float((approximation[stable] - score_error[stable]).max())
            uncertain |= stable & (approximation + score_error >= best_lower)
        if constant and uncertain.any():
            # All windows have the same index, choose deterministically.
            y, x = map(
                int,
                np.unravel_index(
                    np.argmin(np.where(eligible, counts, len(scores))), grid.shape
                ),
            )
            uncertain.fill(False)
            uncertain[y, x] = True
        for raw_y, raw_x in zip(*np.nonzero(uncertain), strict=True):
            y, x = int(raw_y), int(raw_x)
            rows = grid.rows(x, y, size)
            keep = np.ones(len(scores), dtype=bool)
            keep[rows] = False
            index = pooled_index(scores, log_weights, keep)
            closest_key = (-direction * index, len(rows), size, y, x)
            if closest is None or closest_key < closest[0]:
                closest = (
                    closest_key,
                    MaskResult(
                        tuple(sorted(map(int, ids[rows]))),
                        index,
                        "infeasible",
                        window=(x, y, size),
                    ),
                )

    if best is not None:
        result = best[1]
        if unknown_at < result.count:
            return MaskResult(
                result.removed_ids, result.index, "feasible", result.window
            )
        return result
    result = closest[1] if closest is not None else MaskResult((), intact, "infeasible")
    if unknown_at < len(scores):
        return MaskResult(result.removed_ids, result.index, "unresolved", result.window)
    return result


def matched_controls(
    tile_ids: ArrayLike,
    grid: TileGrid,
    removed_ids: tuple[int, ...],
    *,
    compact: bool,
    seed: int,
    repeats: int = 20,
    area_tolerance: float = 0.1,
    coverage: Coverage | None = None,
    coverage_tolerance: float = 0.1,
    window_size: int | None = None,
    area_rectangles: ArrayLike | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Draw unique controls matched on count, union area and available masks."""
    ids = _tile_ids(tile_ids, len(grid.rectangles))
    areas = grid.rectangles if area_rectangles is None else _rectangles(area_rectangles)
    if (
        repeats < 0
        or not 0 <= area_tolerance <= 1
        or not 0 <= coverage_tolerance <= 1
        or (window_size is not None and window_size not in grid.sizes)
        or len(areas) != len(ids)
    ):
        raise ValueError("Invalid control count, matching tolerances or window size.")
    source = np.flatnonzero(np.isin(ids, removed_ids))
    if len(source) != len(removed_ids) or len(source) >= len(ids):
        raise ValueError("Control source IDs must be unique and leave a nonempty bag.")
    if not len(source) or not repeats:
        return ()

    rng = np.random.default_rng(seed)
    count = len(source)
    original_area = union_area(areas[source])
    original_coverage = _coverage(coverage, source)
    seen = {tuple(map(int, source))}
    result = []
    if compact:
        candidates = [
            (size, int(y), int(x))
            for size in ((window_size,) if window_size is not None else grid.sizes)
            for y, x in zip(*np.nonzero(grid.counts(size) == count), strict=True)
        ]
        order = rng.permutation(len(candidates))
        draws = (grid.rows(x, y, size) for size, y, x in (candidates[i] for i in order))
    else:
        draws = (
            np.sort(rng.choice(len(ids), count, replace=False))
            for _ in range(max(100, repeats * 20))
        )
    for rows in draws:
        key = tuple(map(int, rows))
        if key in seen:
            continue
        seen.add(key)
        candidate_area = union_area(areas[rows])
        if abs(candidate_area - original_area) > area_tolerance * original_area:
            continue
        candidate_coverage = _coverage(coverage, rows)
        if len(candidate_coverage) != len(original_coverage):
            raise ValueError("Coverage callback returned a different number of values.")
        if any(
            abs(a - b) > coverage_tolerance
            for a, b in zip(candidate_coverage, original_coverage, strict=True)
        ):
            continue
        result.append(tuple(sorted(map(int, ids[rows]))))
        if len(result) == repeats:
            break
    return tuple(result)


def _coverage(coverage: Coverage | None, rows: IntArray) -> tuple[float, ...]:
    values = () if coverage is None else tuple(coverage(tuple(map(int, rows))))
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("Control coverage values must be fractions between 0 and 1.")
    return values
