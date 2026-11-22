"""Report paired validation performance, then evaluate the locked choice once."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.ml.run_resolution_comparison import (
    FITS,
    LEVELS,
    POOLINGS,
    Fit,
    _write_fit_mapping,
    file_digest,
    load_manifest,
    split_rows,
    verify_split,
    write_json,
)


if TYPE_CHECKING:
    from scripts.ml.run_resolution_comparison import (
        ComparisonManifest,
        FitKey,
        FitRecord,
        MappingRow,
    )


METRICS = ("mae", "spearman", "sign_auroc")


class BootstrapInterval(TypedDict):
    ci95: list[float] | None
    defined_samples: int


class SeedSummary(TypedDict):
    mean: float
    sd: float
    seed_scores: list[float]


class PairedSummary(TypedDict):
    seed_deltas: dict[str, list[float]]
    bootstrap: dict[str, BootstrapInterval]


class ValidationReport(TypedDict):
    bundle_digest: str
    validation_slides: int
    validation_bags_at_or_below_100_tiles: int
    validation_scale: dict[str, object] | None
    validation_exclusions: dict[str, str]
    bootstrap_unit: str
    bootstrap_groups: int
    bootstrap_samples: int
    summaries: dict[str, dict[str, SeedSummary]]
    paired: dict[str, PairedSummary]
    fits: dict[str, FitRecord]


class Decision(TypedDict):
    decision: str
    fit: FitKey
    criteria: dict[str, bool]
    l1_gpu_hours: float
    budget_gpu_hours: float


class LockedDecision(Decision):
    bundle_digest: str
    validation_metrics_digest: str
    chosen_fit_record: FitRecord


def metrics(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Score one or more prediction vectors against the same targets."""
    truth_rank = rankdata(target)
    score_rank = rankdata(prediction, axis=-1)
    truth_centered = truth_rank - truth_rank.mean()
    score_centered = score_rank - score_rank.mean(axis=-1, keepdims=True)
    covariance = (score_centered * truth_centered).sum(axis=-1)
    scale = np.linalg.norm(truth_centered) * np.linalg.norm(score_centered, axis=-1)
    spearman = np.divide(
        covariance, scale, out=np.full_like(covariance, np.nan), where=scale > 0
    )
    positive = target >= 0
    num_pos = int(positive.sum())
    num_neg = len(positive) - num_pos
    auc = np.full_like(covariance, np.nan)
    if num_pos and num_neg:
        # Average ranks handle tied predictions as half a positive/negative win.
        auc = (score_rank[..., positive].sum(axis=-1) - num_pos * (num_pos + 1) / 2) / (
            num_pos * num_neg
        )
    mae = np.abs(prediction - target).mean(axis=-1)
    return np.stack((mae, spearman, auc), axis=-1)


def paired_intervals(
    target: np.ndarray,
    scores: np.ndarray,
    group_ids: list[str],
    samples: int,
) -> dict[str, BootstrapInterval]:
    indices: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        indices.setdefault(group, []).append(index)
    groups = tuple(indices.values())
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(samples):
        draw = rng.integers(0, len(groups), size=len(groups))
        subset = np.concatenate([groups[group] for group in draw])
        resampled = metrics(target[subset], scores[..., subset])
        # First dimension is resolution (L1, L2). The remaining fit axis is seed.
        deltas.append((resampled[0] - resampled[1]).mean(axis=0))
    replicates = np.asarray(deltas)
    intervals: dict[str, BootstrapInterval] = {}
    for metric, series in zip(METRICS, replicates.T, strict=True):
        defined = series[np.isfinite(series)]
        intervals[metric] = {
            "ci95": np.quantile(defined, [0.025, 0.975]).tolist()
            if defined.size
            else None,
            "defined_samples": len(defined),
        }
    return intervals


def _fit_record(
    output: Path, fit: Fit, bundle_digest: str
) -> tuple[FitRecord, Path, Path]:
    directory = output / fit.name
    record: FitRecord = json.loads((directory / "fit.json").read_text())
    if record["fit"] != fit.key() or record["bundle_digest"] != bundle_digest:
        raise ValueError(f"{fit.name} does not match the comparison manifest")
    checkpoint = directory / record["checkpoint"]
    config = directory / record["config"]
    if file_digest(checkpoint) != record["checkpoint_digest"]:
        raise ValueError(f"Checkpoint has changed for {fit.name}")
    if file_digest(config) != record["config_digest"]:
        raise ValueError(f"Fit configuration has changed for {fit.name}")
    return record, checkpoint, config


def _predict_frame(
    fit: Fit,
    split: str,
    directory: Path,
    rows: dict[str, MappingRow],
    checkpoint: Path,
    config_path: Path,
) -> pd.DataFrame:
    from hydra.utils import instantiate
    from lightning.pytorch import Trainer
    from omegaconf import OmegaConf

    from ml.data.datamodule import DataModule

    config = OmegaConf.load(config_path)
    if config.test_after_fit or "test" in config.datamodule:
        raise ValueError(f"Fit configuration allowed automatic testing for {fit.name}")
    with tempfile.TemporaryDirectory(prefix="resolution-scoring-") as workdir:
        mapping = Path(workdir) / "mapping.csv"
        _write_fit_mapping(mapping, list(rows.values()))
        split_config = OmegaConf.create(
            {
                "_target_": config.datamodule.val._target_,
                "split": split,
                "levels": {fit.level: {"uris": {split: str(directory)}}},
                "data_mapping": str(mapping),
                "label_mode": "index",
            }
        )
        batch_size = config.datamodule.batch_size
        if split == "val":
            data = DataModule(batch_size=batch_size, val=split_config)
            data.setup("validate")
            loader = data.val_dataloader()
        elif split == "test":
            data = DataModule(batch_size=batch_size, test=split_config)
            data.setup("test")
            loader = data.test_dataloader()
        else:
            raise ValueError(f"Unsupported scoring split: {split}")
        model = instantiate(config.ml)
        trainer = Trainer(devices=1, logger=False, enable_checkpointing=False)
        predictions = trainer.predict(
            model,
            dataloaders=loader,
            ckpt_path=str(checkpoint),
            return_predictions=True,
        )

    scores: dict[str, float] = {}
    for batch in predictions or ():
        if not isinstance(batch, dict):
            raise RuntimeError(f"Expected a prediction batch dictionary for {fit.name}")
        values = batch["predictions"].detach().cpu().reshape(-1).numpy()
        for metadata, score in zip(batch["metadata"], values, strict=True):
            slide_id = metadata["slide_id"]
            if slide_id in scores or not math.isfinite(score):
                raise ValueError(
                    f"Duplicate slide or non-finite score for {fit.name}: {slide_id}"
                )
            scores[slide_id] = float(score)
    if scores.keys() != rows.keys():
        raise ValueError(
            f"Prediction cohort differs for {fit.name}: {sorted(scores.keys() ^ rows.keys())}"
        )
    return pd.DataFrame(
        {
            "slide_id": slide_id,
            "target": float(rows[slide_id]["mammaprint_index"]),
            "prediction": scores[slide_id],
        }
        for slide_id in sorted(rows)
    )


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _group_ids(
    rows: dict[str, MappingRow], patient_column: str | None
) -> tuple[str, list[str]]:
    ids = sorted(rows)
    if patient_column is None:
        patient_column = next(
            (
                column
                for column in ("patient_id", "patient")
                if all(rows[slide].get(column, "").strip() for slide in ids)
            ),
            None,
        )
    if patient_column is None:
        return "slide", ids
    groups = [rows[slide].get(patient_column, "").strip() for slide in ids]
    if not all(groups):
        raise ValueError(
            f"Patient grouping column is missing or incomplete: {patient_column}"
        )
    return patient_column, groups


def _decision(report: ValidationReport, args: argparse.Namespace) -> Decision:
    summaries = report["summaries"]
    best = {
        level: min(
            POOLINGS,
            key=lambda pooling: summaries[f"l{level}/{pooling}"]["mae"]["mean"],
        )
        for level in LEVELS
    }
    l1 = summaries[f"l1/{best[1]}"]
    l2 = summaries[f"l2/{best[2]}"]
    ci = report["paired"]["mean"]["bootstrap"]["mae"]["ci95"]
    checks = {
        "primary_mae_ci_below_zero": bool(ci and ci[1] < 0),
        "best_l1_mae_improvement_at_least_2_percent": (
            l2["mae"]["mean"] > 0 and l1["mae"]["mean"] <= 0.98 * l2["mae"]["mean"]
        ),
        "best_l1_spearman_drop_at_most_0_02": (
            l2["spearman"]["mean"] - l1["spearman"]["mean"] <= 0.02
        ),
        "best_l1_sign_auroc_drop_at_most_0_02": (
            l2["sign_auroc"]["mean"] - l1["sign_auroc"]["mean"] <= 0.02
        ),
        "within_embedding_gpu_hour_budget": args.gpu_hours <= args.budget_gpu_hours,
    }
    level = 1 if all(checks.values()) else 2
    return {
        "decision": "go" if level == 1 else "no-go",
        "fit": Fit(level, best[level], 0).key(),
        "criteria": checks,
        "l1_gpu_hours": args.gpu_hours,
        "budget_gpu_hours": args.budget_gpu_hours,
    }


def _markdown(report: ValidationReport, decision: Decision) -> str:
    lines = [
        "# L1 vs L2 validation",
        "",
        f"Each fit uses {report['validation_slides']} validation slides and training seed 0, 1, or 2.",
        f"The matched cohort excludes {len(report['validation_exclusions'])} slides. "
        f"It includes {report['validation_bags_at_or_below_100_tiles']} bags with at most 100 tiles.",
        "",
        "| Resolution | Pooling | MAE | Spearman | Sign-AUROC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for setting, summary in report["summaries"].items():
        level, pooling = setting.split("/")
        cells = [f"{summary[k]['mean']:.4f} ± {summary[k]['sd']:.4f}" for k in METRICS]
        lines.append(f"| {level.upper()} | {pooling} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            f"The paired bootstrap resamples by {report['bootstrap_unit']} across both resolutions and all seeds.",
            "Intervals are conditional on the trained seeds.",
            "",
            "| Pooling | Metric | Seed 0 | Seed 1 | Seed 2 | Mean delta | 95% paired CI |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pooling, paired in report["paired"].items():
        for key in METRICS:
            deltas = paired["seed_deltas"][key]
            ci = paired["bootstrap"][key]["ci95"]
            interval = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "undefined"
            cells = [f"{delta:.4f}" for delta in [*deltas, np.mean(deltas)]]
            lines.append(f"| {pooling} | {key} | {' | '.join(cells)} | {interval} |")
    fit = Fit(**decision["fit"])
    lines.extend(
        [
            "",
            f"Decision **{decision['decision']}**. Locked checkpoint: `{fit.name}`.",
            "",
        ]
    )
    lines.extend(
        f"- {name.replace('_', ' ')}: {'pass' if OK else 'fail'}"
        for name, OK in decision["criteria"].items()
    )
    lines.extend(
        [
            "",
            f"L1 used {decision['l1_gpu_hours']:g} GPU-hours against a budget of {decision['budget_gpu_hours']:g}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validation(args: argparse.Namespace, manifest: ComparisonManifest) -> None:
    decision_path = args.output / "decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Decision already locked in {decision_path}")
    rows = split_rows(args.data_mapping, manifest, "val")
    frames = {}
    records = {fit: _fit_record(args.output, fit, manifest["digest"]) for fit in FITS}
    directories = {
        level: verify_split(args.bundle, level, "val", manifest["splits"]["val"])
        for level in LEVELS
    }
    for fit, (_, checkpoint, config) in records.items():
        frame = _predict_frame(
            fit, "val", directories[fit.level], rows, checkpoint, config
        )
        _write_predictions(args.output / fit.name / "validation.parquet", frame)
        frames[fit] = frame

    target = np.array(
        [float(rows[slide_id]["mammaprint_index"]) for slide_id in sorted(rows)]
    )
    if not np.isfinite(target).all():
        raise ValueError("Validation contains non-finite MammaPrint targets")
    unit, group_ids = _group_ids(rows, args.patient_column)
    summaries: dict[str, dict[str, SeedSummary]] = {}
    paired: dict[str, PairedSummary] = {}
    for pooling in POOLINGS:
        scores = np.stack(
            [
                np.stack(
                    [
                        frames[fit]["prediction"].to_numpy()
                        for fit in FITS
                        if fit.level == level and fit.pooling == pooling
                    ]
                )
                for level in LEVELS
            ]
        )
        values = metrics(target, scores)
        if not np.isfinite(values).all():
            raise ValueError(
                f"A {pooling} validation metric is undefined. Check ranks and class coverage."
            )
        for index, level in enumerate(LEVELS):
            summaries[f"l{level}/{pooling}"] = {
                metric: {
                    "mean": floated.mean().item(),
                    "sd": floated.std(ddof=1).item(),
                    "seed_scores": floated.tolist(),
                }
                for metric, floated in zip(METRICS, values[index].T, strict=True)
            }
        paired[pooling] = {
            "seed_deltas": dict(
                zip(METRICS, (values[0] - values[1]).T.tolist(), strict=True)
            ),
            "bootstrap": paired_intervals(
                target, scores, group_ids, args.bootstrap_samples
            ),
        }
    report: ValidationReport = {
        "bundle_digest": manifest["digest"],
        "validation_slides": len(rows),
        "validation_bags_at_or_below_100_tiles": sum(
            count <= 100 for count in manifest["splits"]["val"].values()
        ),
        "validation_scale": manifest.get("realized_scale", {}).get("val"),
        "validation_exclusions": manifest.get("exclusions", {}).get("val", {}),
        "bootstrap_unit": unit,
        "bootstrap_groups": len(set(group_ids)),
        "bootstrap_samples": args.bootstrap_samples,
        "summaries": summaries,
        "paired": paired,
        "fits": {fit.name: record[0] for fit, record in records.items()},
    }
    decision = _decision(report, args)
    metrics_path = args.output / "validation_metrics.json"
    write_json(metrics_path, report)
    (args.output / "validation.md").write_text(_markdown(report, decision))
    chosen = Fit(**decision["fit"])
    locked: LockedDecision = {
        **decision,
        "bundle_digest": manifest["digest"],
        "validation_metrics_digest": file_digest(metrics_path),
        "chosen_fit_record": records[chosen][0],
    }
    write_json(decision_path, locked)
    print(f"Saved validation report and locked {chosen.name}")


def _test(args: argparse.Namespace, manifest: ComparisonManifest) -> None:
    decision: LockedDecision = json.loads((args.output / "decision.json").read_text())
    if decision["bundle_digest"] != manifest["digest"] or decision[
        "validation_metrics_digest"
    ] != file_digest(args.output / "validation_metrics.json"):
        raise ValueError(
            "The test input or validation report differs from the locked decision"
        )
    fit = Fit(**decision["fit"])
    if fit.seed != 0 or fit not in FITS:
        raise ValueError("The locked fit is outside the experiment protocol")
    record, checkpoint, config = _fit_record(args.output, fit, manifest["digest"])
    if record != decision["chosen_fit_record"]:
        raise ValueError("The chosen fit differs from the locked decision")

    marker = args.output / "test_started.json"
    if marker.exists():
        raise RuntimeError(
            "The locked test has already been attempted. Inspect the existing test evidence."
        )
    rows = split_rows(args.data_mapping, manifest, "test")
    directory = verify_split(
        args.bundle,
        fit.level,
        "test",
        manifest["splits"]["test"],
        digests=manifest["files"][f"l{fit.level}"]["test"],
    )
    try:
        with marker.open("x") as stream:
            json.dump(decision, stream, indent=2)
    except FileExistsError as error:
        raise RuntimeError(
            "The locked test has already been attempted. Inspect the existing test evidence."
        ) from error
    frame = _predict_frame(fit, "test", directory, rows, checkpoint, config)
    _write_predictions(args.output / "test_predictions.parquet", frame)
    target = frame["target"].to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("Test contains non-finite MammaPrint targets")
    scores = metrics(target, frame["prediction"].to_numpy())
    result = {
        "bundle_digest": manifest["digest"],
        "fit": fit.key(),
        "mlflow_run_id": record["mlflow_run_id"],
        "slides": len(frame),
        "metrics": {
            key: score.item() if np.isfinite(score) else None
            for key, score in zip(METRICS, scores, strict=True)
        },
    }
    write_json(args.output / "test_metrics.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    validation = subcommands.add_parser("validation")
    test = subcommands.add_parser("test")
    for subcommand in (validation, test):
        subcommand.add_argument("--bundle", type=Path, required=True)
        subcommand.add_argument("--data-mapping", type=Path, required=True)
        subcommand.add_argument("--output", type=Path, required=True)
    validation.add_argument("--gpu-hours", type=float, required=True)
    validation.add_argument("--budget-gpu-hours", type=float, required=True)
    validation.add_argument(
        "--patient-column",
        help="Group validation bootstrap draws by this mapping column",
    )
    validation.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.mode == "validation" and (
        args.bootstrap_samples < 1
        or not math.isfinite(args.gpu_hours)
        or not math.isfinite(args.budget_gpu_hours)
        or args.gpu_hours < 0
        or args.budget_gpu_hours <= 0
    ):
        parser.error(
            "Provide a positive bootstrap sample count, nonnegative GPU-hours, and a positive budget"
        )
    for key in ("bundle", "data_mapping", "output"):
        setattr(args, key, getattr(args, key).resolve())
    manifest = load_manifest(args.bundle, args.data_mapping)
    with (args.output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "validation":
            _validation(args, manifest)
        else:
            _test(args, manifest)


if __name__ == "__main__":
    main()
