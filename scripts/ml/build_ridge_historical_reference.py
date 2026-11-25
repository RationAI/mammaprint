"""Verify archived L2 cm5 validation checkpoints and build a ridge reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore[import-untyped]
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts, list_artifacts, load_text
from mlflow.utils.uri import append_to_uri_path
from torch.nn.functional import linear


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SOURCE = "l2/cancer_mask_5"
WIDTH = 2560
CONFIG = "configs/config-resolved.yaml"
CHECKPOINT = "checkpoints/best/checkpoint.ckpt"
MONITOR = "valid/loss"
MAE = "valid/mae"
F1 = "valid/thresholded_f1"
SPLITS = ("train", "val")
TOLERANCE = 1e-5


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cache_inputs(
    cache: Path, mapping: Path
) -> tuple[dict[str, Any], Path, dict[str, tuple[str, int | None]]]:
    manifest = json.loads((cache / "manifest.json").read_text())
    digest = manifest.pop("digest")
    body = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))
    require(hashlib.sha256(body.encode()).hexdigest() == digest, "Invalid cache digest")
    require(manifest["pilot_slides"] is None, "A full cache is required")
    require(
        sha(mapping) == manifest["mapping_sha256"], "Mapping differs from the cache"
    )
    source = manifest["sources"][SOURCE]
    require(source["status"] == "available", "L2 cm5 cache source is unavailable")
    path = f"stats/dev/{SOURCE}.parquet"
    require(
        sha(cache / path) == source["files"][path], "L2 cm5 cache checksum mismatch"
    )
    coverage_path = cache / "coverage.parquet"
    require(
        sha(coverage_path) == manifest["coverage_sha256"],
        "Cache coverage checksum mismatch",
    )
    coverage_rows = pq.read_table(
        coverage_path,
        columns=["slide_id", "status", "tile_count"],
        filters=[("source", "=", SOURCE), ("split", "in", list(SPLITS))],
    ).to_pylist()
    coverage = {
        row["slide_id"]: (row["status"], row["tile_count"]) for row in coverage_rows
    }
    require(len(coverage) == len(coverage_rows), "Duplicate L2 cm5 coverage rows")
    return source, cache / path, coverage


def model_and_uris(
    config: Mapping[str, Any], mapping: Path, *, include_all: bool = True
) -> dict[str, str]:
    targets = {
        "encoder": "ml.models.encoders.IdentityEncoder",
        "aggregator": "ml.models.aggregators.MeanPool",
        "head": "ml.models.heads.LinearHead",
        "output_activation": "torch.nn.Identity",
        "loss": "torch.nn.MSELoss",
    }
    ml = config["ml"]
    for name, target in targets.items():
        require(ml[name]["_target_"] == target, f"Unexpected archived {name}")
    require(
        config["label_mode"] == "index"
        and config["feature_dim"] == WIDTH
        and ml["head"]["out_dim"] == 1
        and ml["head"]["in_dim"] == WIDTH,
        "Expected scalar index regression on 2560-dimensional embeddings",
    )
    metric = ml["metrics"]["thresholded_f1"]
    require(
        metric["_target_"] == "ml.metrics.thresholded_f1"
        and float(metric["threshold"]) == 0
        and metric.get("target_threshold") in (None, 0),
        "Historical F1 uses a different threshold",
    )
    checkpoint = config["trainer"]["callbacks"]["model_checkpoint"]
    require(
        checkpoint["monitor"] == MONITOR
        and checkpoint["mode"] == "min"
        and checkpoint["save_top_k"] == 1
        and checkpoint["filename"] == "best",
        "Unexpected historical checkpoint selection",
    )
    levels = config["dataset"]["levels"]
    require(len(levels) == 1, "Expected one embedding level")
    level = levels.get("2", levels.get(2))
    require(level is not None, "Expected level 2")
    sources = ("all", *SPLITS) if include_all else SPLITS
    uris = {key: level["uris"][key] for key in sources}
    for split in SPLITS:
        dataset = config["datamodule"][split]
        embedded = dataset["levels"]
        level = embedded.get("2", embedded.get(2))
        require(
            dataset["_target_"] == "ml.data.SingleScaleDataset"
            and dataset["split"] == split
            and dataset["label_mode"] == "index"
            and Path(dataset["data_mapping"]) == mapping
            and dataset.get("train_tile_cap") is None
            and level is not None
            and level["uris"][split] == uris[split],
            f"Archived {split} dataset differs from the expected full embedding bags",
        )
    return uris


def mapping_rows(path: Path) -> dict[str, tuple[str, str, float | None]]:
    rows: dict[str, tuple[str, str, float | None]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["path"]:
                continue
            slide, split, record = (
                Path(row["path"]).stem,
                row["split"],
                row["record_num"].strip(),
            )
            require(
                slide not in rows and bool(record) and split in (*SPLITS, "test"),
                "Invalid mapping membership",
            )
            target = None
            if split in SPLITS:
                try:
                    value = float(row["mammaprint_index"])
                    target = float(np.float32(value)) if math.isfinite(value) else None
                except (TypeError, ValueError):
                    pass
            rows[slide] = (split, record, target)
    return rows


def split_coverage(
    uris: Mapping[str, str],
    inventory: Mapping[str, Any],
    rows: Mapping[str, tuple[str, str, float | None]],
    cached_count: Mapping[str, int],
    coverage: Mapping[str, tuple[str, int | None]],
) -> dict[str, list[str]]:
    rosters = {}
    for split in SPLITS:
        files = [
            entry
            for entry in list_artifacts(artifact_uri=uris[split])
            if not entry.is_dir
        ]
        parquets = [entry for entry in files if Path(entry.path).suffix == ".parquet"]
        require(bool(parquets), f"No Parquets in historical {split} artifact")
        seen = set()
        for entry in parquets:
            slide = Path(entry.path).stem
            require(
                slide not in seen and slide in rows and rows[slide][0] == split,
                f"Invalid historical {split} membership",
            )
            seen.add(slide)
            expected = inventory.get(slide)
            require(
                expected is not None
                and expected["name"] == Path(entry.path).name
                and expected["bytes"] is not None
                and expected["bytes"] == entry.file_size,
                f"Historical {split} file differs from source inventory: {slide}",
            )
        expected_ids = {
            slide for slide in inventory if slide in rows and rows[slide][0] == split
        }
        require(
            seen == expected_ids, f"Historical {split} and source slide lists differ"
        )
        eligible = []
        for slide in sorted(seen):
            if rows[slide][2] is None:
                continue
            if coverage.get(slide) == ("empty", 0) and slide not in cached_count:
                continue
            count = cached_count.get(slide)
            require(
                count is not None
                and count > 0
                and coverage.get(slide) == ("ok", count),
                f"Labeled historical {split} slide lacks verified cache coverage: {slide}",
            )
            eligible.append(slide)
        require(bool(eligible), f"No labeled, nonempty historical {split} slides")
        rosters[split] = eligible
    dev_records = {row[1] for row in rows.values() if row[0] in SPLITS}
    test_records = {row[1] for row in rows.values() if row[0] == "test"}
    require(not dev_records & test_records, "A record crosses the test boundary")
    return rosters


def select_metric_step(
    checkpoint: Mapping[str, Any], history: Mapping[str, Sequence[Any]]
) -> dict[str, float | int]:
    loss_history = history[MONITOR]
    require(bool(loss_history), "Missing historical validation loss")
    best = min(loss_history, key=lambda score: (score.value, score.step))
    scores = {
        float(item["best_model_score"])
        for item in checkpoint.get("callbacks", {}).values()
        if item.get("monitor") == MONITOR and item.get("best_model_score") is not None
    }
    require(
        checkpoint["global_step"] - 1 == best.step
        and len(scores) == 1
        and math.isclose(next(iter(scores)), best.value, rel_tol=1e-6, abs_tol=1e-8),
        f"Checkpoint step {checkpoint['global_step']} does not match the logged minimum-loss step {best.step}",
    )
    measures: dict[str, float | int] = {
        "logged_step": best.step,
        "validation_mse": best.value,
    }
    for key, name in ((MAE, "mae"), (F1, "f1")):
        values = {item.value for item in history[key] if item.step == best.step}
        require(
            len(values) == 1 and math.isfinite(next(iter(values))),
            f"Missing or ambiguous {key} at best step",
        )
        measures[name] = next(iter(values))
    return measures


def reference_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = pred - target
    a, b = pred >= 0, target >= 0
    true_positive = int(np.count_nonzero(a & b))
    positives = int(np.count_nonzero(a) + np.count_nonzero(b))
    return {
        "mae": float(np.abs(errors).mean()),
        "validation_mse": float(np.square(errors).mean()),
        "f1": 2 * true_positive / positives if positives else 0.0,
    }


def raw_probes(
    val_uri: str,
    roster: list[str],
    count: Mapping[str, int],
    means: Mapping[str, np.ndarray],
    weight: torch.Tensor,
    bias: torch.Tensor,
    scratch: Path,
) -> tuple[dict[str, float], float]:
    order = sorted(roster, key=lambda slide: (count[slide], slide))
    probe = [
        order[index]
        for index in sorted(set(np.linspace(0, len(order) - 1, 5, dtype=int)))
    ]
    predictions = {}
    errors = []
    for slide in probe:
        with tempfile.TemporaryDirectory(prefix="ridge-probe-", dir=scratch) as dest:
            path = download_artifacts(
                artifact_uri=append_to_uri_path(val_uri, slide + ".parquet"),
                dst_path=dest,
            )
            vectors = pq.read_table(path, columns=["embedding"])[
                "embedding"
            ].combine_chunks()
            lengths = vectors.value_lengths().to_numpy(zero_copy_only=False)
            require(
                len(vectors) == count[slide] and bool(np.all(lengths == WIDTH)),
                "Archived validation probe tile count or width differs",
            )
            bag = np.array(
                vectors.flatten().to_numpy(zero_copy_only=False), dtype=np.float32
            )
            bag = bag.reshape(len(vectors), WIDTH)
            output = float(
                linear(torch.from_numpy(bag).mean(dim=0), weight, bias).item()
            )
            cached = float(
                means[slide].astype(np.float64) @ weight.numpy()[0].astype(np.float64)
                + bias.item()
            )
            require(
                math.isclose(cached, output, rel_tol=TOLERANCE, abs_tol=TOLERANCE),
                "Raw-bag and cache predictions differ",
            )
            predictions[slide] = output
            errors.append(abs(cached - output))
    return predictions, max(errors)


def build(
    cache: Path, mapping: Path, run_ids: Sequence[str], scratch: Path
) -> dict[str, Any]:
    source, path, coverage = cache_inputs(cache, mapping)
    table = pq.read_table(path, columns=["slide_id", "tile_count", "mean_raw"])
    ids = table["slide_id"].to_pylist()
    require(len(ids) == len(set(ids)), "Duplicate cache slides")
    vectors = (
        table["mean_raw"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    )
    means = dict(zip(ids, vectors.reshape(len(ids), WIDTH), strict=True))
    counts = dict(zip(ids, table["tile_count"].to_pylist(), strict=True))
    rows = mapping_rows(mapping)
    client = MlflowClient()
    archived = []
    split_uris = None
    for run_id in run_ids:
        require(bool(re.fullmatch(r"[a-f0-9]{32}", run_id)), "Invalid run ID")
        run = client.get_run(run_id)
        require(
            run.info.status == "FINISHED", f"Historical run is not finished: {run_id}"
        )
        config_uri = f"runs:/{run_id}/{CONFIG}"
        config_text = load_text(config_uri)
        config = yaml.safe_load(config_text)
        uris = model_and_uris(config, mapping)
        require(
            uris["all"] == source["uri"],
            "Archived source does not match the full cache",
        )
        require(
            split_uris is None or uris == split_uris,
            "Historical runs use different split artifacts",
        )
        split_uris = uris
        archived.append((run, config, config_uri, config_text))
    if split_uris is None:
        raise ValueError("Provide at least one archived run")
    rosters = split_coverage(split_uris, source["inventory"], rows, counts, coverage)
    x = np.stack([means[slide] for slide in rosters["val"]])
    y = np.array([rows[slide][2] for slide in rosters["val"]], dtype=np.float32)

    accepted = []
    excluded = []
    parity = None
    for run, config, config_uri, config_text in archived:
        run_id = run.info.run_id
        uri = f"runs:/{run_id}/{CHECKPOINT}"
        with tempfile.TemporaryDirectory(prefix="ridge-archive-", dir=scratch) as dest:
            path = Path(download_artifacts(artifact_uri=uri, dst_path=dest))
            digest = sha(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        history = {
            key: client.get_metric_history(run_id, key) for key in (MONITOR, MAE, F1)
        }
        try:
            reviewed = select_metric_step(checkpoint, history)
        except ValueError as exc:
            excluded.append(
                {"run_id": run_id, "seed": config["seed"], "reason": str(exc)}
            )
            print(f"Excluded seed {config['seed']}: {exc}", file=sys.stderr)
            continue
        state = checkpoint["state_dict"]
        require(
            set(state) == {"head.linear.weight", "head.linear.bias"},
            "Unexpected checkpoint state",
        )
        weight, bias = state["head.linear.weight"], state["head.linear.bias"]
        require(
            tuple(weight.shape) == (1, WIDTH) and tuple(bias.shape) == (1,),
            "Invalid checkpoint head shape",
        )
        require(
            bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()),
            "Nonfinite checkpoint head",
        )
        scores = reference_scores(x @ weight.numpy()[0] + bias.item(), y)
        for key, value in scores.items():
            require(
                math.isclose(
                    value, reviewed[key], rel_tol=TOLERANCE, abs_tol=TOLERANCE
                ),
                f"{run_id} recomputed {key}={value:.8g} differs from checkpoint metric {reviewed[key]:.8g}",
            )
        predictions, error = raw_probes(
            split_uris["val"], rosters["val"], counts, means, weight, bias, scratch
        )
        revision = run.data.tags.get("mlflow.source.git.commit")
        require(bool(revision), "Missing archive source revision")
        accepted.append(
            {
                "run_id": run_id,
                "seed": config["seed"],
                "code_revision": revision,
                "config": config_uri,
                "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "checkpoint": uri,
                "checkpoint_sha256": digest,
                "checkpoint_selection": "minimum valid/loss",
                "epoch": checkpoint["epoch"],
                **reviewed,
                "recomputed_validation": scores,
                "raw_probe_count": len(predictions),
                "max_raw_cache_prediction_error": error,
            }
        )
        if parity is None:
            parity = {
                "weights": weight.numpy()[0].tolist(),
                "intercept": bias.item(),
                "predictions": predictions,
            }
    require(bool(accepted), "No archived checkpoint passed verification")
    groups = {split: {rows[slide][1] for slide in rosters[split]} for split in SPLITS}
    return {
        "source_uri": split_uris["all"],
        "train_uri": split_uris["train"],
        "val_uri": split_uris["val"],
        "train_slides": rosters["train"],
        "val_slides": rosters["val"],
        "transform": "raw",
        "runs": accepted,
        "parity": parity,
        "verification": {
            "excluded_runs": excluded,
            "train_records": len(groups["train"]),
            "val_records": len(groups["val"]),
            "train_val_record_overlap": len(groups["train"] & groups["val"]),
            "mapping_sha256": sha(mapping),
            "development_stats_sha256": source["files"][f"stats/dev/{SOURCE}.parquet"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(len(set(args.run_id)) == len(args.run_id), "Duplicate archived run ID")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    torch.set_num_threads(1)
    result = build(args.cache, args.data_mapping, args.run_id, args.scratch)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(
            args.output,
            "x",
            encoding="utf-8",
            opener=lambda name, flags: os.open(name, flags, 0o600),
        ) as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
