"""Grouped nested cross-validation on cached slide statistics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing.slide_statistics import digest, file_digest


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from preprocessing.slide_statistics import Slide


type FloatArray = NDArray[np.float64]
type Indices = NDArray[np.intp]
type Split = tuple[Indices, Indices]
type Block = tuple[str, str]

BASELINE = "l2_cm5_mean"
RAW_BASELINE = "l2_cm5_mean_raw"
CORE = (
    BASELINE,
    "l3_cm5_mean",
    "l4_cm5_mean",
    "l5_cm5_mean",
    "l2_cm5_mean_std",
    "l2_l3_cm5_mean",
)
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
METRICS = ("mae", "spearman", "sign_auroc")
STATS = {"mean_ln", "std_ln", "mean_raw", "std_raw"}


@dataclass(frozen=True)
class Feature:
    name: str
    blocks: tuple[Block, ...]


def features_from_config(config: Mapping[str, Any]) -> dict[str, Feature]:
    features = {}
    for name, blocks in config["features"].items():
        parsed = []
        for source, column in blocks:
            parts = source.split("/")
            if (
                len(parts) != 2
                or parts[0] not in {"l1", "l2", "l3", "l4", "l5"}
                or not parts[1].replace("_", "").isalnum()
                or column not in STATS
            ):
                raise ValueError(
                    f"Invalid feature block for {name}: {(source, column)}"
                )
            parsed.append((source, column))
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError(f"Expected nonempty, unique blocks for {name}")
        features[name] = Feature(name, tuple(parsed))
    return features


class FeatureStore:
    """Load one cache partition and verify each selected block at first access."""

    def __init__(self, path: Path, manifest: Mapping[str, Any], partition: str) -> None:
        if partition not in {"dev", "test"}:
            raise ValueError(f"Invalid cache partition: {partition}")
        self.path = path
        self.manifest = manifest
        self.partition = partition
        self.blocks: dict[Block, dict[str, FloatArray]] = {}

    def has(self, feature: Feature) -> bool:
        for source, _ in feature.blocks:
            name = f"stats/{self.partition}/{source}.parquet"
            if name not in self.manifest["sources"].get(source, {}).get("files", {}):
                return False
        return True

    def load(self, block: Block) -> dict[str, FloatArray]:
        if block not in self.blocks:
            source, column = block
            name = f"stats/{self.partition}/{source}.parquet"
            path = self.path / name
            expected = self.manifest["sources"][source]["files"][name]
            if file_digest(path) != expected:
                raise ValueError(f"Cache partition checksum mismatch: {path}")
            table = pq.read_table(path, columns=["slide_id", column])
            ids = table["slide_id"].to_pylist()
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate slide IDs in {path}")
            values = table[column].combine_chunks()
            if values.null_count:
                raise ValueError(f"Null cached feature in {path}: {column}")
            flat = values.flatten().to_numpy(zero_copy_only=False)
            matrix = np.asarray(flat, dtype=np.float64).reshape(
                len(ids), values.type.list_size
            )
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite cache features in {path}: {column}")
            self.blocks[block] = dict(zip(ids, matrix, strict=True))
        return self.blocks[block]

    def eligible(self, feature: Feature) -> set[str]:
        return set.intersection(*(set(self.load(block)) for block in feature.blocks))

    def matrix(self, feature: Feature, ids: Sequence[str]) -> FloatArray:
        matrices = [self.load(block) for block in feature.blocks]
        return np.concatenate(
            [np.stack([matrix[slide] for slide in ids]) for matrix in matrices], axis=1
        )


@dataclass(frozen=True)
class Cohort:
    ids: tuple[str, ...]
    groups: NDArray[np.str_]
    y: FloatArray

    @cached_property
    def key(self) -> str:
        return digest(self.ids)

    @classmethod
    def from_slides(cls, ids: Iterable[str], slides: Mapping[str, Slide]) -> Cohort:
        ids = tuple(sorted(ids))
        return cls(
            ids,
            np.asarray([slides[slide].record_num for slide in ids], dtype=str),
            np.asarray([slides[slide].target for slide in ids], dtype=np.float64),
        )


@dataclass(frozen=True)
class Protocol:
    outer_seeds: tuple[int, ...]
    outer_folds: int
    inner_folds: int
    alphas: tuple[float, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Protocol:
        protocol = cls(
            tuple(config.get("outer_seeds", range(5))),
            int(config.get("outer_folds", 5)),
            int(config.get("inner_folds", 3)),
            tuple(
                sorted(
                    float(alpha)
                    for alpha in config.get("alphas", [10.0**k for k in range(-6, 7)])
                )
            ),
        )
        if (
            not protocol.outer_seeds
            or len(set(protocol.outer_seeds)) != len(protocol.outer_seeds)
            or any(
                not isinstance(seed, int) or not 0 <= seed < 2**32
                for seed in protocol.outer_seeds
            )
            or min(protocol.outer_folds, protocol.inner_folds) < 2
            or not protocol.alphas
            or len(set(protocol.alphas)) != len(protocol.alphas)
            or any(not math.isfinite(alpha) or alpha <= 0 for alpha in protocol.alphas)
        ):
            raise ValueError("Invalid grouped CV seed, fold, or alpha configuration")
        return protocol


class InfeasibleCVError(ValueError):
    """The available grouped classes cannot support the requested protocol."""


def derived_seed(*values: int) -> int:
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def grouped_splits(
    y: FloatArray,
    groups: NDArray[np.str_],
    n_splits: int,
    seed: int,
) -> tuple[Split, ...]:
    signs = y >= 0
    counts = [len(set(groups[signs == sign])) for sign in (False, True)]
    if min(counts) >= n_splits:
        for attempt in range(32):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed if not attempt else derived_seed(seed, attempt),
            )
            splits = tuple(splitter.split(signs, signs, groups))
            if all(
                signs[fit].any()
                and not signs[fit].all()
                and signs[assess].any()
                and not signs[assess].all()
                and not (set(groups[fit]) & set(groups[assess]))
                for fit, assess in splits
            ):
                return splits
    raise InfeasibleCVError(
        f"Cannot construct {n_splits} grouped folds (seed {seed}) with both signs. "
        f"Records={len(set(groups))}, negative/nonnegative record counts={counts}."
    )


def training_prefixes(
    cohort: Cohort, train: Indices, seed: int
) -> dict[float, Indices]:
    by_record: dict[str, list[int]] = {}
    for index in train:
        by_record.setdefault(cohort.groups[index], []).append(int(index))
    strata: tuple[list[str], list[str]] = ([], [])
    for record, indices in by_record.items():
        nonnegative = np.count_nonzero(cohort.y[indices] >= 0)
        strata[int(nonnegative * 2 >= len(indices))].append(record)
    for stratum in strata:
        stratum.sort(
            key=lambda record: (
                hashlib.sha256(f"{seed}:{record}".encode()).digest(),
                record,
            )
        )
    return {
        fraction: np.asarray(
            sorted(
                index
                for stratum in strata
                for record in stratum[: math.ceil(fraction * len(stratum))]
                for index in by_record[record]
            ),
            dtype=np.intp,
        )
        for fraction in FRACTIONS
    }


@dataclass(frozen=True)
class Fold:
    seed: int
    number: int
    fraction: float
    train: Indices
    assess: Indices
    inner: tuple[Split, ...]


def nested_folds(
    cohort: Cohort, protocol: Protocol, *, learning_curve: bool = False
) -> tuple[Fold, ...]:
    folds = []
    for seed in protocol.outer_seeds:
        for number, (train, assess) in enumerate(
            grouped_splits(cohort.y, cohort.groups, protocol.outer_folds, seed)
        ):
            fit_seed = derived_seed(seed, number)
            subsets = (
                training_prefixes(cohort, train, fit_seed)
                if learning_curve
                else {1.0: train}
            )
            for fraction, subset in subsets.items():
                try:
                    inner = grouped_splits(
                        cohort.y[subset],
                        cohort.groups[subset],
                        protocol.inner_folds,
                        fit_seed,
                    )
                except InfeasibleCVError as exc:
                    raise InfeasibleCVError(
                        f"Outer seed {seed}, fold {number}, fraction {fraction:g}: {exc}"
                    ) from exc
                folds.append(Fold(seed, number, fraction, subset, assess, inner))
    return tuple(folds)


def ridge_path(
    x: FloatArray, y: FloatArray, alphas: Sequence[float]
) -> tuple[FloatArray, FloatArray]:
    """Return one unscaled ridge solution per column, with unpenalized intercepts."""
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    left, singular, right = np.linalg.svd(x - x_mean, full_matrices=False)
    sigma = singular[:, None]
    weights = right.T @ ((left.T @ (y - y_mean))[:, None] * sigma / (sigma**2 + alphas))
    return weights, y_mean - x_mean @ weights


def fit_ridge(
    x: FloatArray, y: FloatArray, splits: Sequence[Split], alphas: Sequence[float]
) -> tuple[float, FloatArray, float]:
    errors = []
    for train, assess in splits:
        weights, bias = ridge_path(x[train], y[train], alphas)
        scores = x[assess] @ weights + bias
        errors.append(np.abs(scores - y[assess, None]).mean(axis=0))
    mean_errors = np.mean(errors, axis=0)
    index = min(range(len(alphas)), key=lambda i: (mean_errors[i], -alphas[i]))
    alpha = alphas[index]
    weights, bias = ridge_path(x, y, (alpha,))
    return alpha, weights[:, 0], float(bias[0])


def score_metrics(y: FloatArray, pred: FloatArray) -> dict[str, float | str | None]:
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Expected nonempty finite targets and predictions")
    signs = y >= 0
    constant = np.ptp(y) == 0 or np.ptp(pred) == 0
    single_sign = signs.all() or not signs.any()
    return {
        "mae": float(np.abs(y - pred).mean()),
        "spearman": None if constant else float(spearmanr(y, pred).statistic),
        "spearman_reason": "constant target or prediction" if constant else None,
        "sign_auroc": None if single_sign else float(roc_auc_score(signs, pred)),
        "sign_auroc_reason": "only one target sign" if single_sign else None,
    }


def evaluate(
    cohort: Cohort, x: FloatArray, fold: Fold, alphas: Sequence[float]
) -> dict[str, Any]:
    train, assess = fold.train, fold.assess
    alpha, weights, bias = fit_ridge(x[train], cohort.y[train], fold.inner, alphas)
    return {
        "seed": fold.seed,
        "fold": fold.number,
        "fraction": fold.fraction,
        "alpha": alpha,
        "fit_slides": len(train),
        "fit_records": len(set(cohort.groups[train])),
        "assess_slides": len(assess),
        "assess_records": len(set(cohort.groups[assess])),
        "assessment_digest": digest([cohort.ids[i] for i in assess]),
        **score_metrics(cohort.y[assess], x[assess] @ weights + bias),
    }


def fold_rows(
    scope: str, cohort: Cohort, folds: Sequence[Fold]
) -> Iterable[dict[str, Any]]:
    for fold in folds:
        parts: list[tuple[str, int | None, Indices]] = [
            ("outer_fit", None, fold.train),
            ("outer_assess", None, fold.assess),
        ]
        for inner, (fit, assess) in enumerate(fold.inner):
            parts.extend(
                [
                    ("inner_fit", inner, fold.train[fit]),
                    ("inner_assess", inner, fold.train[assess]),
                ]
            )
        for role, inner_number, indices in parts:
            for index in indices:
                yield {
                    "screen": scope,
                    "cohort": cohort.key,
                    "seed": fold.seed,
                    "fold": fold.number,
                    "fraction": fold.fraction,
                    "role": role,
                    "inner_fold": inner_number,
                    "slide_id": cohort.ids[index],
                    "record_num": cohort.groups[index],
                }


def summarize(
    values: Sequence[float], ratios: Sequence[float]
) -> tuple[float | None, float | None, int]:
    count = len(values)
    mean = float(np.mean(values)) if count else None
    se = (
        float(np.std(values, ddof=1) * np.sqrt(1 / count + np.mean(ratios)))
        if count > 1
        else None
    )
    return mean, se, count


def summary_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for metric in METRICS:
        valid = [row for row in rows if row[metric] is not None]
        mean, se, count = summarize(
            [row[metric] for row in valid],
            [row["assess_records"] / row["fit_records"] for row in valid],
        )
        summary.update({metric: mean, f"{metric}_se": se, f"{metric}_n": count})
    return summary


def paired_rows(
    rows: Sequence[dict[str, Any]], ref: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    key = itemgetter("seed", "fold", "assessment_digest")
    targets = {key(row): row for row in ref}
    if (
        len(targets) != len(ref)
        or set(map(key, rows)) != targets.keys()
        or len(rows) != len(ref)
    ):
        raise ValueError(
            "Paired comparisons require identical outer fold and slide assignments"
        )
    contrasts = []
    for metric in METRICS:
        differences = []
        ratios = []
        for row in rows:
            other = targets[key(row)]
            if row[metric] is not None and other[metric] is not None:
                differences.append(row[metric] - other[metric])
                ratios.append(
                    row["assess_records"]
                    / min(row["fit_records"], other["fit_records"])
                )
        mean, se, count = summarize(differences, ratios)
        contrasts.append(
            {"metric": metric, "delta": mean, "se": se, "valid_folds": count}
        )
    return contrasts
