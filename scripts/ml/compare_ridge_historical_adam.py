"""Export development L2 cm5 features or compare ridge and Adam on its original split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SOURCE = "l2/cancer_mask_5"
WIDTH = 2560
DEVELOPMENT = {"train", "val"}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_development(cache: Path, mapping: Path) -> None:
    """Write a development-only NPZ bundle to stdout from a read-only mount."""
    import pyarrow.parquet as pq

    manifest = json.loads((cache / "manifest.json").read_text())
    digest = manifest.pop("digest")
    body = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if (
        hashlib.sha256(body.encode()).hexdigest() != digest
        or manifest["pilot_slides"] is not None
    ):
        raise ValueError("Invalid full cache digest")
    mapping_sha = file_sha(mapping)
    if mapping_sha != manifest["mapping_sha256"]:
        raise ValueError("Mapping does not match the cache")
    filename = f"stats/dev/{SOURCE}.parquet"
    source = manifest["sources"][SOURCE]
    stats_sha = file_sha(cache / filename)
    if stats_sha != source["files"][filename]:
        raise ValueError("L2 cm5 development statistics checksum mismatch")

    with mapping.open(encoding="utf-8-sig", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            if row["split"] not in DEVELOPMENT or not row["path"]:
                continue
            slide = Path(row["path"]).stem
            if slide in rows or not row["record_num"].strip():
                raise ValueError(f"Invalid record membership for {slide}")
            try:
                target = float(row["mammaprint_index"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(target):
                rows[slide] = row["split"], row["record_num"].strip(), target
    table = pq.read_table(cache / filename, columns=["slide_id", "mean_raw"])
    ids = table["slide_id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cached slide ID")
    values = table["mean_raw"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    features = values.reshape(len(ids), WIDTH)
    keep = [i for i, slide in enumerate(ids) if slide in rows]
    selected = [ids[i] for i in keep]
    np.savez_compressed(
        sys.stdout.buffer,
        source_uri=source["uri"],
        mapping_sha256=mapping_sha,
        cache_digest=digest,
        development_stats_sha256=stats_sha,
        ids=np.asarray(selected),
        splits=np.asarray([rows[slide][0] for slide in selected]),
        records=np.asarray([rows[slide][1] for slide in selected]),
        targets=np.asarray([rows[slide][2] for slide in selected], dtype=np.float64),
        mean_raw=features[keep],
    )


def fit_adam(
    x: np.ndarray, y: np.ndarray, alpha: float, steps: int, learning_rate: float
) -> tuple[np.ndarray, float]:
    import torch

    x_mean, y_mean = x.mean(axis=0), y.mean()
    features = torch.from_numpy(x - x_mean)
    targets = torch.from_numpy(y - y_mean)
    weights = torch.zeros(x.shape[1], dtype=features.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate / 1000
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (features @ weights - targets).square().mean()
        loss = loss + (alpha / len(y)) * weights.square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
    fitted = weights.detach().numpy()
    return fitted, float(y_mean - x_mean @ fitted)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    from ml.ridge_screening import score_metrics

    return {
        **score_metrics(y, pred),
        "mse": float(np.square(y - pred).mean()),
        "f1": float(f1_score(y >= 0, pred >= 0, zero_division=0)),
    }


def compare(
    reference: Path, bundle: Path, steps: int, learning_rate: float
) -> dict[str, Any]:
    from ml.ridge_screening import Protocol, fit_ridge, grouped_splits

    historical = json.loads(reference.read_text())
    if historical["transform"] != "raw" or len(historical["runs"]) != 1:
        raise ValueError("Expected one verified historical raw-mean Adam checkpoint")
    with np.load(bundle, allow_pickle=False) as data:
        if (
            data["source_uri"].item() != historical["source_uri"]
            or data["mapping_sha256"].item()
            != historical["verification"]["mapping_sha256"]
            or data["development_stats_sha256"].item()
            != historical["verification"]["development_stats_sha256"]
            or not set(data["splits"]).issubset(DEVELOPMENT)
        ):
            raise ValueError("Export does not match historical development data")
        ids: dict[str, int] = {slide: i for i, slide in enumerate(data["ids"])}
        if len(ids) != len(data["ids"]):
            raise ValueError("Duplicate slide ID")
        train, val = (
            np.array([ids[slide] for slide in sorted(historical[f"{split}_slides"])])
            for split in ("train", "val")
        )
        if (
            not (data["splits"][train] == "train").all()
            or not (data["splits"][val] == "val").all()
            or set(data["records"][train]) & set(data["records"][val])
        ):
            raise ValueError("Historical split membership changed")
        features = data["mean_raw"].astype(np.float64)
        targets, groups = data["targets"], data["records"][train]
        cache_digest = data["cache_digest"].item()
        stats_sha = data["development_stats_sha256"].item()

    parity = historical["parity"]
    archived_weights = np.asarray(parity["weights"], dtype=np.float64)
    archived_bias = float(parity["intercept"])
    probe = parity["predictions"]
    expected = np.asarray(list(probe.values()), dtype=np.float64)
    actual = (
        features[[ids[slide] for slide in probe]] @ archived_weights + archived_bias
    )
    if not len(probe) or not np.allclose(expected, actual, rtol=1e-5, atol=1e-5):
        raise ValueError("Historical checkpoint parity failed")

    y_train, y_val = targets[train], targets[val]
    x_train, x_val = features[train], features[val]
    protocol = Protocol.from_config({})
    inner = grouped_splits(
        y_train, groups, protocol.inner_folds, protocol.outer_seeds[0]
    )
    alpha, ridge_w, ridge_b = fit_ridge(x_train, y_train, inner, protocol.alphas)
    adam_w, adam_b = fit_adam(x_train, y_train, alpha, steps, learning_rate)

    def evaluate(w: np.ndarray, b: float) -> dict[str, Any]:
        objective = (
            np.square(x_train @ w + b - y_train).sum() + alpha * np.square(w).sum()
        ) / len(y_train)
        return {
            **metrics(y_val, x_val @ w + b),
            "training_objective_at_ridge_alpha": float(objective),
        }

    result = {
        "train_slides": len(train),
        "validation_slides": len(val),
        "historical_run_id": historical["runs"][0]["run_id"],
        "cache_digest": cache_digest,
        "development_stats_sha256": stats_sha,
        "reference_sha256": file_sha(reference),
        "alpha": alpha,
        "matched_adam_steps": steps,
        "matched_adam_learning_rate": learning_rate,
        "ridge": evaluate(ridge_w, ridge_b),
        "historical_adam": evaluate(archived_weights, archived_bias),
        "matched_adam": evaluate(adam_w, adam_b),
    }
    for key, name in (("mae", "mae"), ("mse", "validation_mse"), ("f1", "f1")):
        if not math.isclose(
            result["historical_adam"][key],
            historical["runs"][0][name],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(f"Historical validation {key} did not reproduce")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    export = modes.add_parser("export")
    export.add_argument("--cache", type=Path, required=True)
    export.add_argument("--data-mapping", type=Path, required=True)
    evaluation = modes.add_parser("compare")
    evaluation.add_argument("--reference", type=Path, required=True)
    evaluation.add_argument("--bundle", type=Path, required=True)
    evaluation.add_argument("--steps", type=int, default=2000)
    evaluation.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.mode == "export":
        export_development(args.cache, args.data_mapping)
    else:
        import torch
        from threadpoolctl import threadpool_limits

        if args.steps < 1 or not 0 < args.learning_rate < math.inf:
            parser.error("Expected positive Adam steps and learning rate")
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            result = compare(
                args.reference, args.bundle, args.steps, args.learning_rate
            )
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
