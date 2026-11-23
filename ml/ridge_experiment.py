"""Orchestrate the cache-backed ridge experiment and its held-out evaluation."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
from importlib.metadata import version
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

from ml.ridge_report import test_section, write_report
from ml.ridge_screening import (
    BASELINE,
    CORE,
    METRICS,
    RAW_BASELINE,
    Cohort,
    Feature,
    FeatureStore,
    InfeasibleCVError,
    Protocol,
    evaluate,
    features_from_config,
    fit_ridge,
    fold_rows,
    grouped_splits,
    nested_folds,
    paired_rows,
    score_metrics,
    summary_rows,
)
from preprocessing.slide_statistics import (
    digest,
    file_digest,
    load_manifest,
    load_membership,
    write_json,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from preprocessing.slide_statistics import Slide


L2_CM5 = "l2/cancer_mask_5"
CURVE = "learning-curve"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def signed_write(path: Path, value: dict[str, Any]) -> None:
    write_json(path, {**value, "digest": digest(value)})


def signed_read(path: Path) -> dict[str, Any]:
    value = read_json(path)
    body = {key: item for key, item in value.items() if key != "digest"}
    if value.get("digest") != digest(body):
        raise ValueError(f"Invalid digest in {path}")
    return value


class Experiment:
    def __init__(self, config: Mapping[str, Any], *, partition: str = "dev") -> None:
        self.config = config
        self.cache_dir = config["cache_dir"]
        self.manifest = load_manifest(self.cache_dir)
        if self.manifest.get("pilot") or self.manifest.get("pilot_slides"):
            raise ValueError(
                "A pilot cache is for sizing only. Use a completed full cache."
            )
        self.mapping = config["data_mapping"]
        self.mapping_digest = file_digest(self.mapping)
        if self.mapping_digest != self.manifest["mapping_sha256"]:
            raise ValueError("Mapping checksum differs from the cache manifest")
        self.protocol = Protocol.from_config(config)
        self.store = FeatureStore(self.cache_dir, self.manifest, partition)

    def development(self) -> dict[str, Slide]:
        return {
            key: row
            for key, row in load_membership(self.mapping, labels="dev").items()
            if row.split != "test"
        }

    def labeled_development(self) -> dict[str, Slide]:
        return {
            key: row
            for key, row in self.development().items()
            if row.target is not None
        }

    def source_uri(self) -> str:
        return self.manifest["sources"][L2_CM5]["uri"]

    def identity(self) -> dict[str, Any]:
        return {
            "cache_digest": self.manifest["digest"],
            "mapping_sha256": self.mapping_digest,
        }

    def settings(self) -> dict[str, Any]:
        return {
            "outer_seeds": list(self.protocol.outer_seeds),
            "outer_folds": self.protocol.outer_folds,
            "inner_folds": self.protocol.inner_folds,
            "alphas": list(self.protocol.alphas),
            "blas_threads": int(self.config.get("blas_threads", 1)),
            "versions": {
                package: version(package)
                for package in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pyarrow",
                    "threadpoolctl",
                )
            },
        }

    def sanity_settings(self) -> dict[str, Any]:
        return {
            "inner_seed": self.protocol.outer_seeds[0],
            "inner_folds": self.protocol.inner_folds,
            "alphas": list(self.protocol.alphas),
            "f1_margin": float(self.config.get("sanity_f1_margin", 0.02)),
            "mae_relative_margin": float(self.config.get("sanity_mae_margin", 0.05)),
        }


def verify_historical(
    experiment: Experiment, reference: Mapping[str, Any], slides: Mapping[str, Slide]
) -> tuple[Feature, dict[str, Cohort], dict[str, Any]]:
    if reference.get("source_uri") != experiment.source_uri():
        raise ValueError("Historical reference source URI differs from cached L2 cm5")
    if any(
        not isinstance(reference.get(key), str) or not reference[key].strip()
        for key in ("train_uri", "val_uri")
    ):
        raise ValueError(
            "Historical reference must identify both original split artifacts"
        )
    transform = reference.get("transform")
    if transform not in {"raw", "ln"}:
        raise ValueError("Historical transform must be 'raw' or 'ln'")
    feature = Feature("historical", ((L2_CM5, f"mean_{transform}"),))
    available = experiment.store.eligible(feature)
    cohorts = {}
    for split in ("train", "val"):
        roster = reference[f"{split}_slides"]
        if not roster or len(set(roster)) != len(roster):
            raise ValueError(f"Expected nonempty, unique historical {split} slide IDs")
        invalid = [
            slide
            for slide in roster
            if slide not in slides
            or slides[slide].split != split
            or slide not in available
        ]
        if invalid:
            raise ValueError(
                f"Historical {split} slides missing, mislabeled, or in wrong split: {invalid}"
            )
        cohorts[split] = Cohort.from_slides(roster, slides)

    runs = reference["runs"]
    required = (
        "run_id",
        "seed",
        "code_revision",
        "config",
        "checkpoint",
        "checkpoint_selection",
    )
    if not runs or len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("Expected distinct historical run IDs")
    for run in runs:
        if any(run.get(key) is None or run[key] == "" for key in required):
            raise ValueError("Historical runs need seed and checkpoint provenance")
        if not 0 <= float(run["f1"]) <= 1 or not 0 <= float(run["mae"]) < math.inf:
            raise ValueError(
                "Historical F1 must be in [0, 1] and MAE must be nonnegative and finite"
            )

    parity = reference["parity"]
    probe = tuple(sorted(parity["predictions"]))
    if not probe or not set(probe).issubset(cohorts["val"].ids):
        raise ValueError(
            "Parity probes must come from the historical validation roster"
        )
    x = experiment.store.matrix(feature, probe)
    weights = np.asarray(parity["weights"], dtype=np.float64)
    expected = np.asarray(
        [parity["predictions"][slide] for slide in probe], dtype=np.float64
    )
    if (
        weights.shape != (x.shape[1],)
        or not np.isfinite(weights).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Invalid parity coefficients or predictions")
    bias = float(parity["intercept"])
    if not math.isfinite(bias):
        raise ValueError("Invalid parity intercept")
    actual = x @ weights + bias
    parity_result = {
        "slide_ids": probe,
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "passed": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-5)),
    }
    return feature, cohorts, parity_result


def run_sanity(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    exp = Experiment(config)
    reference_path = config["historical_reference"]
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    matched, cohorts, parity = verify_historical(
        exp, reference, exp.labeled_development()
    )
    train, val = cohorts["train"], cohorts["val"]
    splits = grouped_splits(
        train.y, train.groups, exp.protocol.inner_folds, exp.protocol.outer_seeds[0]
    )
    features = [matched]
    if reference["transform"] == "raw":
        features.append(Feature("layernorm_sensitivity", ((L2_CM5, "mean_ln"),)))
    results = {}
    for feature in features:
        x = exp.store.matrix(feature, train.ids)
        alpha, weights, bias = fit_ridge(x, train.y, splits, exp.protocol.alphas)
        pred = exp.store.matrix(feature, val.ids) @ weights + bias
        results[feature.name] = {
            "alpha": alpha,
            "f1": float(f1_score(val.y >= 0, pred >= 0, zero_division=0)),
            **score_metrics(val.y, pred),
        }
    historical = {
        "f1_min": min(float(run["f1"]) for run in reference["runs"]),
        "f1_max": max(float(run["f1"]) for run in reference["runs"]),
        "mae_mean": float(np.mean([float(run["mae"]) for run in reference["runs"]])),
    }
    settings = exp.sanity_settings()
    f1_margin = settings["f1_margin"]
    mae_margin = settings["mae_relative_margin"]
    if not 0 <= f1_margin <= 1 or not 0 <= mae_margin < math.inf:
        raise ValueError("Invalid historical sanity margins")
    within_tolerance = results["historical"]["f1"] >= historical[
        "f1_min"
    ] - f1_margin and results["historical"]["mae"] <= historical["mae_mean"] * (
        1 + mae_margin
    )
    resolution = reference.get("resolution", "").strip()
    outcome = {
        "passed": parity["passed"] and (within_tolerance or bool(resolution)),
        "within_tolerance": within_tolerance,
        "resolution": resolution,
        "settings": settings,
        "source_uri": exp.source_uri(),
        "l2_dev_sha256": exp.manifest["sources"][L2_CM5]["files"][
            f"stats/dev/{L2_CM5}.parquet"
        ],
        "transform": reference["transform"],
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference": {key: reference[key] for key in ("train_uri", "val_uri", "runs")},
        "cohorts": {
            split: {
                "slides": len(cohort.ids),
                "slide_ids_digest": cohort.key,
                "records": len(set(cohort.groups)),
            }
            for split, cohort in cohorts.items()
        },
        "record_overlap": sorted(set(train.groups) & set(val.groups)),
        "parity": parity,
        "historical": historical,
        "ridge": results,
        **exp.identity(),
    }
    output.mkdir(parents=True, exist_ok=True)
    signed_write(output / "sanity.json", outcome)
    return outcome


def read_sanity(exp: Experiment) -> dict[str, Any]:
    sanity = signed_read(exp.config["sanity_dir"] / "sanity.json")
    if (
        sanity["passed"] is not True
        or sanity["source_uri"] != exp.source_uri()
        or sanity["mapping_sha256"] != exp.mapping_digest
        or sanity["reference_sha256"] != file_digest(exp.config["historical_reference"])
        or sanity["settings"] != exp.sanity_settings()
    ):
        raise ValueError(
            "Production screening requires a resolved, matching L2 cm5 sanity check"
        )
    source = exp.manifest["sources"][L2_CM5]
    dev_file = f"stats/dev/{L2_CM5}.parquet"
    if sanity.get("l2_dev_sha256") != source["files"][dev_file]:
        raise ValueError(
            "L2 cm5 features changed following the historical sanity check"
        )
    return sanity


def screen_arms(config: Mapping[str, Any], selected: str) -> dict[str, tuple[str, ...]]:
    screens = {
        name: tuple(dict.fromkeys((BASELINE, *arms)))
        for name, arms in config["screens"].items()
    }
    if selected != "all":
        return {selected: screens[selected]}
    return screens


def match_arms(
    arms: Sequence[str], eligible: Mapping[str, set[str]], slides: Mapping[str, Slide]
) -> Cohort:
    return Cohort.from_slides(
        set.intersection(*(eligible[arm] for arm in arms)), slides
    )


def metrics_fields() -> list[str]:
    return [
        "screen",
        "cohort",
        "feature_set",
        "seed",
        "fold",
        "fraction",
        "alpha",
        "fit_slides",
        "fit_records",
        "assess_slides",
        "assess_records",
        "assessment_digest",
        "mae",
        "spearman",
        "spearman_reason",
        "sign_auroc",
        "sign_auroc_reason",
    ]


def coverage_rows(
    exp: Experiment,
    features: Mapping[str, Feature],
    eligible: Mapping[str, set[str]],
    slides: Mapping[str, Slide],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = exp.cache_dir / "coverage.parquet"
    if file_digest(path) != exp.manifest["coverage_sha256"]:
        raise ValueError("Cache coverage checksum mismatch")
    table = pq.read_table(
        path,
        columns=["source", "slide_id", "status", "reason"],
        filters=[("split", "in", ["train", "val"])],
    )
    reasons = {
        (row["source"], row["slide_id"]): (row["status"], row["reason"])
        for row in table.to_pylist()
    }
    rows = []
    summary = []
    for arm, feature in features.items():
        if arm not in eligible:
            continue
        ids = eligible[arm]
        summary.append(
            {
                "feature_set": arm,
                "slides": len(ids),
                "records": len({slides[slide].record_num for slide in ids}),
            }
        )
        missing_sources = {
            source: set(exp.store.load((source, column)))
            for source, column in feature.blocks
        }
        for slide in sorted(slides.keys() - ids):
            for source, source_ids in missing_sources.items():
                if slide not in source_ids:
                    status, reason = reasons.get((source, slide), ("absent", None))
                    rows.append(
                        {
                            "feature_set": arm,
                            "slide_id": slide,
                            "source": source,
                            "status": status,
                            "reason": reason,
                        }
                    )
    return rows, summary


def run_screen(
    config: Mapping[str, Any], output: Path, selected: str
) -> dict[str, Any]:
    exp = Experiment(config)
    sanity = read_sanity(exp)
    features = features_from_config(config)
    expected_base = Feature(BASELINE, ((L2_CM5, "mean_ln"),))
    if features.get(BASELINE) != expected_base:
        raise ValueError("The reference must be the LayerNormed L2 cm5 mean")
    screens = screen_arms(config, selected)
    development = exp.development()
    slides = {key: row for key, row in development.items() if row.target is not None}
    candidate_names = list(dict.fromkeys(chain.from_iterable(screens.values())))
    if selected == "all" and sanity["transform"] == "raw":
        features[RAW_BASELINE] = Feature(RAW_BASELINE, ((L2_CM5, "mean_raw"),))
        candidate_names.append(RAW_BASELINE)
    candidates = {name: features[name] for name in candidate_names}
    eligible = {
        name: exp.store.eligible(feature) & slides.keys()
        for name, feature in candidates.items()
        if exp.store.has(feature)
    }
    if BASELINE not in eligible:
        raise ValueError("Missing canonical L2 cm5 development cache")
    skipped = {
        name: "source unavailable" for name in candidates if name not in eligible
    }
    selections = {}
    errors = {}
    for name, arms in screens.items():
        usable = tuple(arm for arm in arms if arm in eligible)
        if len(usable) < 2 and name != CURVE:
            errors[name] = "Fewer than two available arms"
            continue
        cohort = match_arms(usable, eligible, slides)
        try:
            folds = nested_folds(cohort, exp.protocol, learning_curve=name == CURVE)
        except InfeasibleCVError as exc:
            errors[name] = str(exc)
            continue
        selections[name] = (usable, cohort, folds)

    ranking_scope = None
    if selected == "all":
        available = tuple(eligible)
        ranking_sets = [("ranking", available)]
        core = tuple(config.get("core_features", CORE))
        if RAW_BASELINE in eligible:
            core = (*core, RAW_BASELINE)
        if core != available and all(arm in eligible for arm in core):
            ranking_sets.append(("ranking-core", core))
        for name, arms in ranking_sets:
            if len(arms) < 2:
                errors[name] = "Fewer than two available arms"
                continue
            cohort = match_arms(arms, eligible, slides)
            try:
                folds = nested_folds(cohort, exp.protocol)
            except InfeasibleCVError as exc:
                errors[name] = str(exc)
                continue
            selections[name] = (arms, cohort, folds)
            ranking_scope = name
            break

    scores = {}
    rows = []
    summaries = []
    contrasts = []
    ranking = []
    for screen, (arms, cohort, folds) in selections.items():
        results = {}
        for arm in arms:
            x = exp.store.matrix(candidates[arm], cohort.ids)
            by_fraction: dict[float, list[dict[str, Any]]] = {}
            for fold in folds:
                key = (cohort.key, arm, fold.seed, fold.number, fold.fraction)
                if key not in scores:
                    scores[key] = evaluate(cohort, x, fold, exp.protocol.alphas)
                row = scores[key]
                rows.append(
                    {"screen": screen, "cohort": cohort.key, "feature_set": arm, **row}
                )
                by_fraction.setdefault(fold.fraction, []).append(row)
            for fraction, values in by_fraction.items():
                results[arm, fraction] = values
                entry = {
                    "screen": screen,
                    "cohort": cohort.key,
                    "feature_set": arm,
                    "fraction": fraction,
                    "slides": len(cohort.ids),
                    "records": len(set(cohort.groups)),
                    "coordinates": x.shape[1],
                    "fit_records_min": min(row["fit_records"] for row in values),
                    "fit_records_max": max(row["fit_records"] for row in values),
                    **summary_rows(values),
                }
                summaries.append(entry)
                if screen == ranking_scope:
                    ranking.append(entry)
        for (arm, fraction), values in results.items():
            ref_fraction = 0.75 if screen == CURVE and fraction == 1 else 1.0
            if (arm, fraction) == (BASELINE, ref_fraction):
                continue
            for contrast in paired_rows(values, results[BASELINE, ref_fraction]):
                contrasts.append(
                    {
                        "screen": screen,
                        "cohort": cohort.key,
                        "feature_set": arm,
                        "fraction": fraction,
                        "reference": BASELINE,
                        "reference_fraction": ref_fraction,
                        **contrast,
                    }
                )

    ranking.sort(key=lambda row: (row["mae"], row["coordinates"], row["feature_set"]))
    exclusions, coverage = coverage_rows(exp, candidates, eligible, slides)
    missing_targets = sorted(development.keys() - slides.keys())
    exclusions.extend(
        {
            "feature_set": "",
            "slide_id": slide,
            "source": "",
            "status": "missing-target",
            "reason": "missing or nonfinite development target",
        }
        for slide in missing_targets
    )
    output.mkdir(parents=True, exist_ok=True)
    # A previous run manifest must never certify a partially replaced result.
    (output / "run.json").unlink(missing_ok=True)
    write_csv(
        output / "folds.csv",
        (
            "screen",
            "cohort",
            "seed",
            "fold",
            "fraction",
            "role",
            "inner_fold",
            "slide_id",
            "record_num",
        ),
        chain.from_iterable(
            fold_rows(name, cohort, folds)
            for name, (_, cohort, folds) in selections.items()
        ),
    )
    write_csv(output / "metrics.csv", metrics_fields(), rows)
    summary_fields = [
        "screen",
        "cohort",
        "feature_set",
        "fraction",
        "slides",
        "records",
        "coordinates",
        "fit_records_min",
        "fit_records_max",
        *chain.from_iterable(
            (metric, f"{metric}_se", f"{metric}_n") for metric in METRICS
        ),
    ]
    write_csv(output / "summary.csv", summary_fields, summaries)
    write_csv(output / "ranking.csv", summary_fields, ranking)
    write_csv(
        output / "contrasts.csv",
        (
            "screen",
            "cohort",
            "feature_set",
            "fraction",
            "reference",
            "reference_fraction",
            "metric",
            "delta",
            "se",
            "valid_folds",
        ),
        contrasts,
    )
    write_csv(
        output / "exclusions.csv",
        ("feature_set", "slide_id", "source", "status", "reason"),
        exclusions,
    )
    write_csv(output / "coverage.csv", ("feature_set", "slides", "records"), coverage)
    run = {
        "screen": selected,
        "settings": exp.settings(),
        "sanity_digest": sanity["digest"],
        "source_uri": exp.source_uri(),
        "feature_blocks": {
            name: feature.blocks for name, feature in candidates.items()
        },
        "ranking_scope": ranking_scope,
        "winner": ranking[0]["feature_set"] if ranking else None,
        "errors": errors,
        "skipped": skipped,
        "missing_development_targets": len(missing_targets),
        "cache_path": str(exp.cache_dir),
        "tables": {
            f"{name}.csv": file_digest(output / f"{name}.csv")
            for name in (
                "folds",
                "metrics",
                "summary",
                "ranking",
                "contrasts",
                "exclusions",
                "coverage",
            )
        },
        **exp.identity(),
    }
    write_report(output, run, sanity, ranking, summaries, contrasts, coverage)
    run["report_sha256"] = file_digest(output / "report.md")
    signed_write(output / "run.json", run)
    return run


def lock_model(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    exp = Experiment(config)
    screen_dir = config["screen_dir"]
    run = signed_read(screen_dir / "run.json")
    sanity = read_sanity(exp)
    if (
        any(run.get(key) != value for key, value in exp.identity().items())
        or run["settings"] != exp.settings()
        or run["sanity_digest"] != sanity["digest"]
        or run["screen"] != "all"
        or not run["winner"]
    ):
        raise ValueError(
            "A matching, ranked all-screens run is required before locking"
        )
    for name, checksum in run["tables"].items():
        if file_digest(screen_dir / name) != checksum:
            raise ValueError(f"Screen table checksum mismatch: {name}")
    feature = Feature(
        run["winner"], tuple(map(tuple, run["feature_blocks"][run["winner"]]))
    )
    report_path = screen_dir / "report.md"
    if file_digest(report_path) != run["report_sha256"]:
        raise ValueError("Development report differs from the ranked screens")
    slides = exp.labeled_development()
    cohort = Cohort.from_slides(exp.store.eligible(feature) & slides.keys(), slides)
    splits = grouped_splits(
        cohort.y, cohort.groups, exp.protocol.inner_folds, exp.protocol.outer_seeds[0]
    )
    x = exp.store.matrix(feature, cohort.ids)
    alpha, weights, bias = fit_ridge(x, cohort.y, splits, exp.protocol.alphas)
    roster = test_roster(load_membership(exp.mapping))
    lock = {
        "feature": feature.name,
        "blocks": feature.blocks,
        "alpha": alpha,
        "coefficients": weights.tolist(),
        "intercept": bias,
        "train_slides": len(cohort.ids),
        "train_records": len(set(cohort.groups)),
        "train_digest": cohort.key,
        "test_roster_digest": digest(roster),
        "test_membership_rule": "original test partition, finite index, complete frozen feature blocks",
        "test_registry": str(config["test_registry"]),
        "metrics": list(METRICS),
        "ranking_digest": run["tables"]["ranking.csv"],
        "screen_digest": run["digest"],
        "report_sha256": run["report_sha256"],
        **exp.identity(),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "lock.json"
    if path.exists() and signed_read(path)["digest"] != digest(lock):
        raise ValueError(f"A different model is already locked at {path}")
    signed_write(path, lock)
    return lock


def test_roster(membership: Mapping[str, Slide]) -> list[tuple[str, str]]:
    return sorted(
        (slide, row.record_num)
        for slide, row in membership.items()
        if row.split == "test"
    )


def predict_test(
    exp: Experiment, lock: Mapping[str, Any], roster: list[tuple[str, str]]
) -> dict[str, Any]:
    membership = load_membership(exp.mapping, labels="test")
    feature = Feature(lock["feature"], tuple(map(tuple, lock["blocks"])))
    eligible = exp.store.eligible(feature)
    ids = []
    exclusions = {}
    for slide, _ in roster:
        if membership[slide].target is None:
            exclusions[slide] = "missing or nonfinite target"
        elif slide not in eligible:
            exclusions[slide] = "missing required feature block"
        else:
            ids.append(slide)
    cohort = Cohort.from_slides(ids, membership)
    if not cohort.ids:
        raise ValueError("No eligible test slides for the frozen feature definition")
    x = exp.store.matrix(feature, cohort.ids)
    pred = x @ np.asarray(lock["coefficients"]) + lock["intercept"]
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite frozen test predictions")
    return {
        "lock_digest": lock["digest"],
        "slides": list(cohort.ids),
        "records": len(set(cohort.groups)),
        "targets": cohort.y.tolist(),
        "predictions": pred.tolist(),
        "exclusions": exclusions,
    }


def evaluate_test(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    lock = signed_read(config["lock_dir"] / "lock.json")
    exp = Experiment(config, partition="test")
    roster = test_roster(load_membership(exp.mapping))
    roster_digest = digest(roster)
    if (
        not roster
        or any(lock.get(key) != value for key, value in exp.identity().items())
        or lock["test_roster_digest"] != roster_digest
        or lock["test_registry"] != str(config["test_registry"])
    ):
        raise ValueError(
            "Frozen model, cache, test cohort, or registry does not match the lock"
        )

    report_path = config["screen_dir"] / "report.md"
    if file_digest(report_path) != lock["report_sha256"]:
        raise ValueError("Development report differs from the frozen lock")
    report = report_path.read_text()
    relative_screen = os.path.relpath(report_path.parent, start=output)
    for name in (
        "learning_curve.png",
        "folds.csv",
        "metrics.csv",
        "summary.csv",
        "contrasts.csv",
        "exclusions.csv",
    ):
        report = report.replace(f"]({name})", f"]({relative_screen}/{name})")

    registry = config["test_registry"] / roster_digest
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (registry / ".evaluation.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        claim = registry / "claim.json"
        if claim.exists() and signed_read(claim)["lock_digest"] != lock["digest"]:
            raise ValueError(
                "This held-out cohort already has a different evaluation claim"
            )
        metrics_path = registry / "test_metrics.json"
        if metrics_path.exists():
            result = signed_read(metrics_path)
            if result["lock_digest"] != lock["digest"]:
                raise ValueError("Saved test result does not match the frozen model")
        else:
            signed_write(claim, {"lock_digest": lock["digest"]})
            payload_path = registry / "predictions.json"
            if payload_path.exists():
                payload = signed_read(payload_path)
            else:
                payload = predict_test(exp, lock, roster)
                signed_write(payload_path, payload)
            if payload["lock_digest"] != lock["digest"]:
                raise ValueError("Saved predictions do not match the frozen model")
            result = {
                "lock_digest": lock["digest"],
                "feature": lock["feature"],
                "expected_slides": len(roster),
                "scored_slides": len(payload["slides"]),
                "scored_records": payload["records"],
                "exclusions": payload["exclusions"],
                **score_metrics(
                    np.asarray(payload["targets"]), np.asarray(payload["predictions"])
                ),
            }
            signed_write(metrics_path, result)
            result = signed_read(metrics_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_metrics.json", result)
    (output / "held_out_report.md").write_text(
        report.rstrip() + "\n\n" + test_section(result)
    )
    return result
