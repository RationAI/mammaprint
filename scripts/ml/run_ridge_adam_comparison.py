"""Run paired single-level ridge and matched-objective Adam CV on development data."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
from threadpoolctl import threadpool_limits

from ml.matched_linear_adam import fit_matched_adam
from ml.ridge_screening import (
    BASELINE,
    METRICS,
    RAW_BASELINE,
    Cohort,
    Feature,
    Protocol,
    features_from_config,
    fit_ridge,
    fold_rows,
    nested_folds,
    summarize,
)
from preprocessing.slide_statistics import digest, file_digest, write_json
from scripts.ml.compare_ridge_historical_adam import metrics
from scripts.ml.run_ridge_screening import config_from_file


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ml.ridge_screening import FloatArray, Fold


SOURCE = "l2/cancer_mask_5"
ALL_METRICS = (*METRICS, "f1", "mse")
OPTIMIZERS = ("ridge", "adam")


@dataclass(frozen=True)
class AdamSettings:
    max_steps: int = 2000
    learning_rate: float = 0.01
    relative_tolerance: float = 1e-8


def candidates(config: Mapping[str, Any]) -> dict[str, Feature]:
    features = features_from_config(config)
    names = dict.fromkeys(
        chain.from_iterable(
            config["screens"][name] for name in ("levels", "statistics", "masks")
        )
    )
    result = {name: features[name] for name in names}
    result[RAW_BASELINE] = Feature(RAW_BASELINE, ((SOURCE, "mean_raw"),))
    if BASELINE not in result or any(
        len({s for s, _ in f.blocks}) != 1 for f in result.values()
    ):
        raise ValueError("Expected the L2 cm5 baseline and single-level features only")
    return result


def read_bundle(
    path: Path,
    features: Mapping[str, Feature],
) -> tuple[Cohort, dict[str, FloatArray], dict[str, Any], list[str], dict[str, int]]:
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(bundle["metadata"].item())
        ids, groups, labels = bundle["ids"], bundle["records"], bundle["targets"]
        if (
            not len(ids)
            or np.any(ids[1:] <= ids[:-1])
            or not set(bundle["splits"]).issubset({"train", "val"})
            or not np.isfinite(labels).all()
        ):
            raise ValueError(
                "Expected unique sorted development slides with finite labels"
            )
        active, unavailable, stats, eligible = {}, [], {}, {}
        for arm, feature in features.items():
            blocks = [f"{source}:{column}" for source, column in feature.blocks]
            if any(block in meta["unavailable"] for block in blocks):
                unavailable.append(arm)
                continue
            for block in blocks:
                if block not in stats:
                    stats[block] = bundle[meta["blocks"][block]]
            eligible[arm] = np.logical_and.reduce(
                [np.isfinite(stats[block]).all(axis=1) for block in blocks]
            )
            active[arm] = blocks
        if not {BASELINE, RAW_BASELINE}.issubset(active):
            raise ValueError(
                "Cached LayerNormed and historical raw L2 cm5 means are required"
            )
        keep = np.logical_and.reduce(list(eligible.values()))
        cohort = Cohort(tuple(ids[keep].tolist()), groups[keep], labels[keep])
        matrices = {
            arm: np.concatenate([stats[block][keep] for block in blocks], axis=1)
            for arm, blocks in active.items()
        }
        coverage = {arm: int(mask.sum()) for arm, mask in eligible.items()}
    return cohort, matrices, meta, unavailable, coverage


def validate_sanity(
    config: Mapping[str, Any], meta: Mapping[str, Any], reference: Path, sanity: Path
) -> None:
    resolved = json.loads(sanity.read_text())
    if (
        resolved["cache_digest"] != meta["cache_digest"]
        or resolved["development_stats_sha256"] != meta["file_sha256"][SOURCE]
        or resolved["reference_sha256"] != file_digest(reference)
    ):
        raise ValueError(
            "Historical sanity comparison does not match the development export"
        )
    ridge, archived = resolved["ridge"], resolved["historical_adam"]
    if ridge["f1"] < archived["f1"] - float(
        config.get("sanity_f1_margin", 0.02)
    ) or ridge["mae"] > archived["mae"] * (
        1 + float(config.get("sanity_mae_margin", 0.05))
    ):
        raise ValueError(
            "The fixed-split ridge comparison did not clear the historical sanity margins"
        )


def one_fold(
    cohort: Cohort,
    features: FloatArray,
    fold: Fold,
    protocol: Protocol,
    settings: AdamSettings,
) -> dict[str, Any]:
    x, y = features[fold.train], cohort.y[fold.train]
    alpha, w, b = fit_ridge(x, y, fold.inner, protocol.alphas)
    optimum = float(
        (np.square(x @ w + b - y).sum() + alpha * np.square(w).sum()) / len(y)
    )
    adam = fit_matched_adam(
        x,
        y,
        alpha,
        optimum,
        max_steps=settings.max_steps,
        lr=settings.learning_rate,
        relative_tolerance=settings.relative_tolerance,
    )
    x_assess, y_assess = features[fold.assess], cohort.y[fold.assess]
    return {
        "seed": fold.seed,
        "fold": fold.number,
        "fraction": fold.fraction,
        "alpha": alpha,
        "fit_records": len(set(cohort.groups[fold.train])),
        "assess_records": len(set(cohort.groups[fold.assess])),
        "fit_slides": len(fold.train),
        "assess_slides": len(fold.assess),
        "assessment_digest": digest([cohort.ids[i] for i in fold.assess]),
        "ridge": {**metrics(y_assess, x_assess @ w + b), "objective": optimum},
        "adam": {
            **metrics(y_assess, x_assess @ adam.weights + adam.intercept),
            "objective": adam.objective,
        },
        "adam_steps": adam.steps,
        "adam_converged": adam.converged,
        "adam_relative_objective_gap": (adam.objective - optimum)
        / max(optimum, np.finfo(float).tiny),
    }


def run_feature(
    name: str,
    features: FloatArray,
    cohort: Cohort,
    folds: Sequence[Fold],
    protocol: Protocol,
    settings: AdamSettings,
    output: Path,
    run_digest: str,
) -> tuple[str, list[dict[str, Any]]]:
    target = output / ".folds" / name
    target.mkdir(parents=True, exist_ok=True)
    records = []
    with threadpool_limits(limits=1):
        for index, fold in enumerate(folds, start=1):
            path = target / f"s{fold.seed}-f{fold.number}-{fold.fraction:g}.json"
            if path.exists():
                row = json.loads(path.read_text())
                expected = {
                    "run_digest": run_digest,
                    "seed": fold.seed,
                    "fold": fold.number,
                    "fraction": fold.fraction,
                    "assessment_digest": digest([cohort.ids[i] for i in fold.assess]),
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"{path} belongs to a different run or fold")
            else:
                row = {
                    **one_fold(cohort, features, fold, protocol, settings),
                    "run_digest": run_digest,
                }
                write_json(path, row)
            records.append(row)
            if index % 5 == 0 or index == len(folds):
                print(f"{name}: {index}/{len(folds)} paired folds", flush=True)
    return name, records


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    ratios = [row["assess_records"] / row["fit_records"] for row in rows]
    for metric in ALL_METRICS:
        valid = [
            (row, ratio)
            for row, ratio in zip(rows, ratios, strict=True)
            if row["ridge"][metric] is not None and row["adam"][metric] is not None
        ]
        ridge = [row["ridge"][metric] for row, _ in valid]
        adam = [row["adam"][metric] for row, _ in valid]
        eligible_ratios = [ratio for _, ratio in valid]
        for name, values in (
            ("ridge", ridge),
            ("adam", adam),
            ("delta", [b - a for a, b in zip(ridge, adam, strict=True)]),
        ):
            mean, se, count = summarize(values, eligible_ratios)
            result[f"{name}_{metric}"] = mean
            result[f"{name}_{metric}_se"] = se
            result[f"{name}_{metric}_n"] = count
    return {
        **result,
        "adam_converged_folds": sum(row["adam_converged"] for row in rows),
        "total_folds": len(rows),
        "adam_median_steps": median([row["adam_steps"] for row in rows]),
        "adam_max_relative_objective_gap": max(
            row["adam_relative_objective_gap"] for row in rows
        ),
        "min_training_records": min(row["fit_records"] for row in rows),
        "max_training_records": max(row["fit_records"] for row in rows),
    }


def write_table(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    iterator = iter(rows)
    first = next(iterator)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(first), lineterminator="\n")
        writer.writeheader()
        writer.writerows(chain((first,), iterator))


def run(args: argparse.Namespace) -> None:
    config = config_from_file(args.config)
    protocol = Protocol.from_config(config)
    settings = AdamSettings(args.steps, args.learning_rate, args.relative_tolerance)
    cohort, matrices, meta, unavailable, coverage = read_bundle(
        args.bundle, candidates(config)
    )
    validate_sanity(config, meta, args.reference, args.sanity)
    folds = nested_folds(cohort, protocol, learning_curve=True)
    full_folds = tuple(fold for fold in folds if fold.fraction == 1)
    planned = {
        "bundle_sha256": file_digest(args.bundle),
        "config_sha256": file_digest(args.config),
        "reference_sha256": file_digest(args.reference),
        "sanity_sha256": file_digest(args.sanity),
        "cache_digest": meta["cache_digest"],
        "available": list(matrices),
        "unavailable": unavailable,
        "coverage": coverage,
        "adam": vars(settings),
        "cohort": cohort.key,
        "slides": len(cohort.ids),
        "records": len(set(cohort.groups)),
        "seeds": protocol.outer_seeds,
        "outer_folds": protocol.outer_folds,
        "inner_folds": protocol.inner_folds,
        "alphas": protocol.alphas,
        "versions": {
            package: version(package) for package in ("numpy", "scipy", "scikit-learn")
        },
        "runner_sha256": file_digest(Path(__file__)),
        "adam_code_sha256": file_digest(
            Path(__file__).resolve().parents[2] / "ml/matched_linear_adam.py"
        ),
        "ridge_code_sha256": file_digest(
            Path(__file__).resolve().parents[2] / "ml/ridge_screening.py"
        ),
    }
    run_digest = digest(planned)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "run.json"
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["digest"] != run_digest
        ):
            raise ValueError(
                "Existing output was made with a different paired comparison"
            )
        write_json(manifest, {**planned, "digest": run_digest})
        write_table(output / "fold_ledger.csv", fold_rows("paired", cohort, folds))
        scores = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [
                pool.submit(
                    run_feature,
                    name,
                    x.astype(np.float64),
                    cohort,
                    folds if name == BASELINE else full_folds,
                    protocol,
                    settings,
                    output,
                    run_digest,
                )
                for name, x in matrices.items()
            ]
            for job in as_completed(jobs):
                name, rows = job.result()
                scores[name] = rows
        write_results(output, scores, planned, run_digest)


def write_results(
    output: Path,
    scores: Mapping[str, list[dict[str, Any]]],
    planned: dict[str, Any],
    run_digest: str,
) -> None:
    raw, rankings, curve = [], [], []
    for arm in planned["available"]:
        values = scores[arm]
        by_fraction: dict[float, list[dict[str, Any]]] = {}
        for row in values:
            by_fraction.setdefault(row["fraction"], []).append(row)
            shared = {
                key: value
                for key, value in row.items()
                if key not in (*OPTIMIZERS, "run_digest")
            }
            for optimizer in OPTIMIZERS:
                raw.append(
                    {
                        "feature_set": arm,
                        "optimizer": optimizer,
                        **shared,
                        **row[optimizer],
                    }
                )
        for fraction, rows in sorted(by_fraction.items()):
            summary = {"feature_set": arm, "fraction": fraction, **aggregate(rows)}
            if fraction == 1:
                rankings.append(summary)
            if arm == BASELINE:
                curve.append(summary)
    rankings.sort(key=lambda row: (row["ridge_mae"], row["feature_set"]))
    write_table(output / "fold_metrics.csv", raw)
    write_table(output / "ranking.csv", rankings)
    write_table(output / "learning_curve.csv", curve)
    write_json(
        output / "results.json",
        {
            "digest": run_digest,
            "ranking": rankings,
            "learning_curve": curve,
            "fold_metric_sha256": file_digest(output / "fold_metrics.csv"),
            "fold_ledger_sha256": file_digest(output / "fold_ledger.csv"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sweeps/ridge_screening/screening.yaml"),
    )
    for name in ("bundle", "reference", "sanity", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--relative-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.steps < 1
        or not 0 < args.learning_rate < math.inf
        or not 0 < args.relative_tolerance < 1
    ):
        parser.error(
            "Expected positive worker count, step budget and valid Adam settings"
        )
    run(args)


if __name__ == "__main__":
    main()
