"""Minimum-mask index bands and closest attainable masks under a deletion cap."""

from __future__ import annotations

from itertools import combinations
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml.explainability.tile_removal import (
    FloatArray,
    IntArray,
    MaskResult,
    MaskStatus,
    _budget,
    _margin_sign,
    _tile_ids,
    _values,
    _weights,
    minimum_class_mask,
    pooled_index,
)


if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

_EXHAUSTIVE_SIZE = 12


def _distance(index: float, lower: float, upper: float) -> float:
    return max(0.0, lower - index, index - upper)


def _in_band(
    scores: FloatArray, logits: FloatArray, lower: float, upper: float
) -> bool | None:
    low = _margin_sign(scores, logits, lower)
    high = _margin_sign(scores, logits, upper)
    if low == -1 or high == 1:
        return False
    if low is None or high is None:
        return None
    return True


def _evaluate(
    scores: FloatArray,
    logits: FloatArray,
    ids: IntArray,
    keep: NDArray[np.bool_],
    lower: float,
    upper: float,
) -> tuple[MaskResult, bool | None]:
    retained_scores, retained_logits = scores[keep], logits[keep]
    membership = _in_band(retained_scores, retained_logits, lower, upper)
    return (
        MaskResult(
            tuple(sorted(map(int, ids[~keep]))),
            pooled_index(retained_scores, retained_logits),
            "feasible" if membership else "unresolved",
        ),
        membership,
    )


def _closer(
    candidate: MaskResult, current: MaskResult, lower: float, upper: float
) -> MaskResult:
    def key(mask: MaskResult) -> tuple[float, int, tuple[int, ...]]:
        return _distance(mask.index, lower, upper), mask.count, mask.removed_ids

    return min(candidate, current, key=key)


def _exhaustive(
    scores: FloatArray,
    logits: FloatArray,
    ids: IntArray,
    lower: float,
    upper: float,
    budget: int,
    deadline: float,
    best: MaskResult,
) -> MaskResult:
    unknown_at = len(scores)
    rows = np.argsort(ids, kind="stable")
    for count in range(1, budget + 1):
        for removed in combinations(map(int, rows), count):
            if monotonic() >= deadline:
                distance = _distance(best.index, lower, upper)
                return MaskResult(
                    best.removed_ids,
                    best.index,
                    "unresolved",
                    distance_bounds=(0.0, distance),
                )
            keep = np.ones(len(rows), dtype=bool)
            keep[list(removed)] = False
            result, membership = _evaluate(scores, logits, ids, keep, lower, upper)
            if membership:
                status: MaskStatus = "minimum" if unknown_at >= count else "feasible"
                return MaskResult(result.removed_ids, result.index, status)
            if membership is None:
                unknown_at = min(unknown_at, count)
            best = _closer(result, best, lower, upper)
    distance = _distance(best.index, lower, upper)
    return MaskResult(
        best.removed_ids,
        best.index,
        "infeasible" if unknown_at == len(scores) else "unresolved",
        distance_bounds=(distance if unknown_at == len(scores) else 0.0, distance),
    )


def _constraint(
    scores: FloatArray, weights: FloatArray, bound: float, direction: int
) -> tuple[FloatArray, float]:
    scale = max(float(np.max(np.abs(scores))), abs(bound), np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        diff = scores - bound
    large = np.isinf(diff)
    diff[large] = scores[large] / scale - bound / scale
    diff[~large] /= scale
    coefficients = direction * diff * weights
    peak = float(np.max(np.abs(coefficients)))
    if peak == 0:
        return coefficients, 0.0
    coefficients /= peak
    dropped = np.abs(coefficients) <= 1e-8
    # Give presolve an outer approximation. HiGHS may ignore tiny matrix
    # entries, so account for their total magnitude explicitly.
    slack = float(np.abs(coefficients[dropped]).sum()) + len(scores) * 1e-12
    coefficients[dropped] = 0.0
    return coefficients, -slack


def _solve(
    scores: FloatArray,
    weights: FloatArray,
    lower: float,
    upper: float,
    budget: int,
    deadline: float,
    *,
    minimize_deletions: bool,
) -> OptimizeResult | None:
    from scipy.optimize import Bounds, LinearConstraint, milp

    remaining = deadline - monotonic()
    if remaining <= 0:
        return None
    ones = np.ones(len(scores), dtype=np.float64)
    lower_row, lower_rhs = _constraint(scores, weights, lower, 1)
    upper_row, upper_rhs = _constraint(scores, weights, upper, -1)
    matrix = np.stack((ones, lower_row, upper_row))
    return milp(
        -ones if minimize_deletions else np.zeros_like(ones),
        integrality=ones,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            matrix, [len(scores) - budget, lower_rhs, upper_rhs], np.inf
        ),
        options={"time_limit": remaining, "mip_rel_gap": 0.0},
    )


def _solution_rows(
    result: OptimizeResult | None, budget: int
) -> NDArray[np.bool_] | None:
    if result is None or result.x is None:
        return None
    rows = np.rint(result.x)
    if not np.all(np.abs(rows - result.x) <= 1e-5):
        return None
    if not np.isin(rows, (0, 1)).all() or int((rows == 0).sum()) > budget:
        return None
    return rows.astype(bool)


def minimum_index_mask(
    scores: ArrayLike,
    log_weights: ArrayLike,
    tile_ids: ArrayLike,
    lower: float,
    upper: float,
    *,
    max_removed: int,
    time_limit: float = 10.0,
    distance_tolerance: float = 1e-4,
) -> MaskResult:
    """Find a minimum deletion into an inclusive band, or bound its shortfall."""
    scores, logits = _values(scores, log_weights)
    ids = _tile_ids(tile_ids, len(scores))
    budget = _budget(max_removed, len(scores))
    status: MaskStatus
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("The target must be an ordered finite interval.")
    if (
        not np.isfinite([time_limit, distance_tolerance]).all()
        or time_limit <= 0
        or distance_tolerance <= 0
    ):
        raise ValueError("Search time and distance tolerance must both be positive.")
    deadline = monotonic() + time_limit
    best = MaskResult((), pooled_index(scores, logits), "unresolved")
    intact = _in_band(scores, logits, lower, upper)
    if intact:
        return MaskResult((), best.index, "minimum")
    if budget == 0:
        distance = _distance(best.index, lower, upper)
        return MaskResult(
            (),
            best.index,
            "unresolved" if intact is None else "infeasible",
            distance_bounds=(0.0 if intact is None else distance, distance),
        )
    if len(scores) <= _EXHAUSTIVE_SIZE:
        result = _exhaustive(scores, logits, ids, lower, upper, budget, deadline, best)
        if intact is None and result.status in ("minimum", "infeasible"):
            status = "feasible" if result.status == "minimum" else "unresolved"
            bounds = (
                None
                if result.distance_bounds is None
                else (0.0, result.distance_bounds[1])
            )
            return MaskResult(
                result.removed_ids, result.index, status, distance_bounds=bounds
            )
        return result

    # Single-bound masks supply verified incumbent candidates even if the
    # integer search cannot finish within its time budget.
    seed_feasible: MaskResult | None = None
    for directed, boundary in ((scores, lower), (-scores, -upper)):
        from_bound = minimum_class_mask(
            directed, logits, ids, "A", threshold=boundary, max_removed=budget
        )
        if from_bound.removed_ids:
            seed_kept = ~np.isin(ids, from_bound.removed_ids)
            candidate, membership = _evaluate(
                scores, logits, ids, seed_kept, lower, upper
            )
            best = _closer(candidate, best, lower, upper)
            if membership and (
                seed_feasible is None
                or (candidate.count, candidate.removed_ids)
                < (seed_feasible.count, seed_feasible.removed_ids)
            ):
                seed_feasible = candidate

    weights = _weights(logits)
    optimization = _solve(
        scores, weights, lower, upper, budget, deadline, minimize_deletions=True
    )
    kept = _solution_rows(optimization, budget)
    if optimization is not None and kept is not None:
        candidate, membership = _evaluate(scores, logits, ids, kept, lower, upper)
        best = _closer(candidate, best, lower, upper)
        if membership:
            status = (
                "minimum"
                if optimization.status == 0 and intact is not None
                else "feasible"
            )
            if seed_feasible is not None and seed_feasible.count < candidate.count:
                return seed_feasible
            return MaskResult(candidate.removed_ids, candidate.index, status)
    if seed_feasible is not None:
        return seed_feasible
    high = _distance(best.index, lower, upper)
    if optimization is None or optimization.status != 2 or intact is None:
        return MaskResult(
            best.removed_ids, best.index, "unresolved", distance_bounds=(0.0, high)
        )

    # Infeasibility of the relaxed constraints excludes the original band.
    # Enclose the nearest feasible expansion using exact checks of incumbents.
    low = 0.0
    while high - low > distance_tolerance:
        delta = (low + high) / 2
        if not np.isfinite([lower - delta, upper + delta]).all():
            break
        expanded = _solve(
            scores,
            weights,
            lower - delta,
            upper + delta,
            budget,
            deadline,
            minimize_deletions=False,
        )
        if expanded is None:
            break
        if expanded.status == 2:
            low = delta
            continue
        kept = _solution_rows(expanded, budget)
        if kept is None:
            break
        candidate, membership = _evaluate(
            scores, logits, ids, kept, lower - delta, upper + delta
        )
        if not membership:
            break
        best = _closer(candidate, best, lower, upper)
        improved = _distance(best.index, lower, upper)
        if improved >= high:
            break
        high = improved
    return MaskResult(
        best.removed_ids, best.index, "infeasible", distance_bounds=(low, high)
    )
