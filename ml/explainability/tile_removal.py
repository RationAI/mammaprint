"""Minimum-cardinality deletions for affine heads over weighted means."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
MaskStatus = Literal["minimum", "feasible", "infeasible", "unresolved"]


@dataclass(frozen=True)
class MaskResult:
    removed_ids: tuple[int, ...]
    index: float
    status: MaskStatus
    window: tuple[int, int, int] | None = None
    distance_bounds: tuple[float, float] | None = None

    @property
    def count(self) -> int:
        return len(self.removed_ids)


def _values(scores: ArrayLike, log_weights: ArrayLike) -> tuple[FloatArray, FloatArray]:
    scores = np.asarray(scores, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or log_weights.shape != scores.shape:
        raise ValueError("Expected equal-length vectors over a nonempty tile bag.")
    if not np.isfinite(scores).all() or not np.isfinite(log_weights).all():
        raise ValueError("Scores and log weights must be finite.")
    return scores, log_weights


def _tile_ids(tile_ids: ArrayLike, size: int) -> IntArray:
    ids = np.asarray(tile_ids)
    if (
        ids.shape != (size,)
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids > np.iinfo(np.int64).max)
        or np.unique(ids).size != size
    ):
        raise ValueError("Tile IDs must be unique nonnegative int64 values.")
    return ids.astype(np.int64, copy=False)


def _budget(max_removed: int | None, size: int) -> int:
    if max_removed is None:
        return size - 1
    if isinstance(max_removed, bool) or not isinstance(max_removed, Integral):
        raise ValueError("The deletion budget must be an integer.")
    if max_removed < 0:
        raise ValueError("The deletion budget must be nonnegative.")
    return min(int(max_removed), size - 1)


def _weights(log_weights: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(log_weights - log_weights.max())


def attention_weights(log_weights: ArrayLike) -> FloatArray:
    """Return original-bag attention weights in the same row order."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if (
        log_weights.ndim != 1
        or not log_weights.size
        or not np.isfinite(log_weights).all()
    ):
        raise ValueError("Expected a nonempty vector of finite attention logits.")
    weights = _weights(log_weights)
    return weights / weights.sum()


def pooled_index(
    scores: ArrayLike, log_weights: ArrayLike, kept: ArrayLike | None = None
) -> float:
    """Recompute the weighted mean using only the supplied retained rows."""
    values, logits = _values(scores, log_weights)
    if kept is not None:
        keep_mask = np.asarray(kept)
        if (
            keep_mask.shape != values.shape
            or keep_mask.dtype.kind != "b"
            or not keep_mask.any()
        ):
            raise ValueError(
                "Expected a boolean keep mask retaining at least one tile."
            )
        values, logits = values[keep_mask], logits[keep_mask]
    weights = _weights(logits)
    weights /= fsum(weights)
    scale = np.max(np.abs(values))
    if scale == 0:
        return 0.0
    scaled = fsum(weights * (values / scale))
    return float(np.clip(scaled, -1.0, 1.0) * scale)


def _margin_terms(
    scores: FloatArray, log_weights: FloatArray, threshold: float
) -> tuple[FloatArray, FloatArray]:
    if not np.isfinite(threshold):
        raise ValueError("The target cutoff must be finite.")
    with np.errstate(over="ignore", divide="ignore"):
        difference = scores - threshold
        magnitudes = np.log(np.abs(difference))
        large = np.isinf(difference)
        if large.any():
            magnitudes[large] = np.log(
                np.abs(scores[large] / 2 - threshold / 2)
            ) + np.log(2.0)
        rel_logits = log_weights - log_weights.max()
    if not np.isfinite(rel_logits).all():
        raise ValueError("The attention logit range exceeds float64 precision.")
    return np.sign(difference), rel_logits + magnitudes


def _margin_error(left: float, right: float, count: int) -> float:
    return np.finfo(np.float64).eps * max(64 * (1 + abs(left) + abs(right)), 8 * count)


def _precise_margin_sign(
    scores: FloatArray, log_weights: FloatArray, threshold: float
) -> int | None:
    """Resolve cancellation with exact equal-weight sums and decimal exponentials."""
    cutoff = Fraction(float(threshold))
    by_logit: dict[float, Fraction] = {}
    for score, logit in zip(scores, log_weights, strict=True):
        margin = Fraction(float(score)) - cutoff
        if margin:
            logit_key = float(logit)
            by_logit[logit_key] = by_logit.get(logit_key, Fraction(0)) + margin
    groups = [(logit, value) for logit, value in by_logit.items() if value]
    if not groups:
        return 0
    if all(value > 0 for _, value in groups):
        return 1
    if all(value < 0 for _, value in groups):
        return -1

    with localcontext() as ctx:
        ctx.prec = 100
        largest = Decimal.from_float(max(logit for logit, _ in groups))
        terms = [
            (Decimal.from_float(logit) - largest).exp()
            * (Decimal(value.numerator) / Decimal(value.denominator))
            for logit, value in groups
        ]
        total = sum(terms, start=Decimal(0))
        tolerance = sum((abs(term) for term in terms), start=Decimal(0)) * Decimal(
            "1e-85"
        )
        if abs(total) <= tolerance:
            return None
        return 1 if total > 0 else -1


def _compare_masses(left: float, right: float, count: int) -> int | None:
    if left == right == -np.inf:
        return 0
    if left == -np.inf:
        return -1
    if right == -np.inf:
        return 1
    if abs(left - right) <= _margin_error(left, right, count):
        return None
    return 1 if left > right else -1


def _margin_sign(
    scores: FloatArray, log_weights: FloatArray, threshold: float
) -> int | None:
    signs, log_masses = _margin_terms(scores, log_weights, threshold)
    left = float(np.logaddexp.reduce(log_masses[signs > 0], initial=-np.inf))
    right = float(np.logaddexp.reduce(log_masses[signs < 0], initial=-np.inf))
    comparison = _compare_masses(left, right, len(scores))
    if comparison is None:
        return _precise_margin_sign(scores, log_weights, threshold)
    return comparison


def _mass_order(
    scores: FloatArray,
    log_weights: FloatArray,
    log_masses: FloatArray,
    ids: IntArray,
    rows: IntArray,
    threshold: float,
    *,
    descending: bool = True,
) -> IntArray:
    # Refine clustered log magnitudes before trusting their order at the cutoff.
    direction = -1 if descending else 1
    order = rows[np.lexsort((ids[rows], direction * log_masses[rows]))]
    ranked = log_masses[order]
    if ranked.size < 2:
        return order
    with np.errstate(over="ignore"):
        gaps = np.abs(np.diff(ranked))
        limits = (
            64
            * np.finfo(np.float64).eps
            * (1 + np.abs(ranked[:-1]) + np.abs(ranked[1:]))
        )
    close = np.flatnonzero(gaps <= limits)
    if not close.size:
        return order

    cuts = np.flatnonzero(np.diff(close) != 1) + 1
    with localcontext() as ctx:
        ctx.prec = 100
        cutoff = Decimal.from_float(float(threshold))
        for chain in np.split(close, cuts):
            begin, end = int(chain[0]), int(chain[-1]) + 2
            group = order[begin:end]
            pairs = {(float(scores[i]), float(log_weights[i])) for i in group}
            if len(pairs) == 1:
                continue
            keys = {
                (score, logit): Decimal.from_float(logit)
                + abs(Decimal.from_float(score) - cutoff).ln()
                for score, logit in pairs
            }
            order[begin:end] = sorted(
                group,
                key=lambda i: (
                    direction * keys[float(scores[i]), float(log_weights[i])],
                    int(ids[i]),
                ),
            )
    return order


def _directed_terms(
    scores: FloatArray,
    log_weights: FloatArray,
    ids: IntArray,
    target_class: str,
    threshold: float,
    protected_ids: tuple[int, ...],
) -> tuple[FloatArray, FloatArray, BoolArray]:
    if target_class not in ("A", "B"):
        raise ValueError("Target class must be A or B.")
    if not np.isin(protected_ids, ids).all():
        raise ValueError("A protected tile ID does not occur in the bag.")
    signs, log_masses = _margin_terms(scores, log_weights, threshold)
    if target_class == "B":
        signs = -signs
    return signs, log_masses, np.isin(ids, protected_ids)


def class_removal_order(
    scores: ArrayLike,
    log_weights: ArrayLike,
    tile_ids: ArrayLike,
    target_class: str,
    *,
    threshold: float = 0.0,
    protected_ids: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Rank deletable tiles by their weighted contribution toward the target."""
    scores, log_weights = _values(scores, log_weights)
    ids = _tile_ids(tile_ids, len(scores))
    signs, log_masses, protected = _directed_terms(
        scores, log_weights, ids, target_class, threshold, protected_ids
    )
    opposed = _mass_order(
        scores,
        log_weights,
        log_masses,
        ids,
        np.flatnonzero((signs < 0) & ~protected),
        threshold,
    )
    neutral = np.sort(ids[(signs == 0) & ~protected])
    favored = _mass_order(
        scores,
        log_weights,
        log_masses,
        ids,
        np.flatnonzero((signs > 0) & ~protected),
        threshold,
        descending=False,
    )
    return tuple(map(int, np.r_[ids[opposed], neutral, ids[favored]]))


def minimum_class_mask(
    scores: ArrayLike,
    log_weights: ArrayLike,
    tile_ids: ArrayLike,
    target_class: str,
    *,
    threshold: float = 0.0,
    max_removed: int | None = None,
    protected_ids: tuple[int, ...] = (),
) -> MaskResult:
    """Find a minimum deletion to reach A (inclusive) or B (strict)."""
    scores, log_weights = _values(scores, log_weights)
    ids = _tile_ids(tile_ids, len(scores))
    budget = _budget(max_removed, len(scores))
    signs, log_masses, protected = _directed_terms(
        scores, log_weights, ids, target_class, threshold, protected_ids
    )

    opposed = signs < 0
    allowed = np.flatnonzero(opposed & ~protected)
    order = _mass_order(scores, log_weights, log_masses, ids, allowed, threshold)
    favor_log = float(np.logaddexp.reduce(log_masses[signs > 0], initial=-np.inf))
    fixed_log = float(
        np.logaddexp.reduce(log_masses[opposed & protected], initial=-np.inf)
    )
    opposed_logs = np.r_[log_masses[order], fixed_log]
    suffix = np.logaddexp.accumulate(opposed_logs[::-1])[::-1]

    # The helpful removals form a monotone prefix. Unknown signs keep the
    # lower bound conservative while we still look for a verified candidate.
    best: int | None = None
    unknown: list[int] = []
    low, high = 0, min(budget, len(order))
    while low <= high:
        count = (low + high) // 2
        comparison = _compare_masses(favor_log, float(suffix[count]), len(scores))
        if comparison is None:
            kept = np.ones(len(scores), dtype=bool)
            kept[order[:count]] = False
            comparison = _precise_margin_sign(
                scores[kept], log_weights[kept], threshold
            )
            if comparison is not None and target_class == "B":
                comparison = -comparison
        if comparison is None:
            unknown.append(count)
            low = count + 1
        elif comparison > 0 or (comparison == 0 and target_class == "A"):
            best = count
            high = count - 1
        else:
            low = count + 1

    status: MaskStatus
    if best is None:
        status = "unresolved" if unknown else "infeasible"
        return MaskResult((), pooled_index(scores, log_weights), status)
    kept = np.ones(len(scores), dtype=bool)
    kept[order[:best]] = False
    status = "feasible" if unknown and min(unknown) < best else "minimum"
    return MaskResult(
        tuple(map(int, ids[order[:best]])),
        pooled_index(scores, log_weights, kept),
        status,
    )
