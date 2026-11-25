"""Score paired archived, fresh, and 90° L2 validation bags without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore[import-untyped]

from preprocessing.embedding_selection import _L2Source, selection_digest
from preprocessing.l2_augmentation import _coordinates, _load_manifest
from preprocessing.resolution_embeddings import (
    FEATURES,
    _selection,
    file_digest,
    is_complete,
)
from preprocessing.tile_augmentation import TileView
from scripts.ml import build_ridge_historical_reference as archive


if TYPE_CHECKING:
    from collections.abc import Iterator


METRICS = (archive.MONITOR, archive.MAE, archive.F1)


def _fresh_config(root: Path, manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    config = json.loads((root / "run.json").read_text())
    view = config.get("tile_view", {})
    seed = view.get("seed")
    if (
        config.get("manifest_digest") != manifest["digest"]
        or config.get("precision") != "fp32"
        or config.get("normalization") != "imagenet-rgb-fp32-v1"
        or not config.get("runtime_id")
        or not re.fullmatch(r"[a-fA-F0-9]{40}", config.get("encoder_revision", ""))
        or "pilot_slides" in config
        or type(seed) is not int
        or view != TileView(kind, seed, "val").provenance()
    ):
        raise ValueError(
            f"Expected a matching production validation {kind} run in {root}"
        )
    expected = {slide["slide_id"] for slide in manifest["slides"]}
    files = {path.stem for path in (root / "embeddings").glob("*.parquet")}
    if files != expected:
        raise ValueError(f"The {kind} slide inventory differs from the frozen probe")
    return config


def _head(
    config_text: str,
    checkpoint_path: Path,
    history: dict[str, Any],
    clean_uri: str,
) -> tuple[dict[str, Any], np.ndarray, float]:
    config = yaml.safe_load(config_text)
    # The archived job's mapped path can differ from this copy of the frozen CSV.
    original_mapping = Path(config["datamodule"]["train"]["data_mapping"])
    if (
        archive.model_and_uris(config, original_mapping, include_all=False)["val"]
        != clean_uri
    ):
        raise ValueError("The archived baseline uses a different validation artifact")
    seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("The archived baseline must record a nonnegative integer seed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selected = archive.select_metric_step(checkpoint, history)
    state = checkpoint["state_dict"]
    if set(state) != {"head.linear.weight", "head.linear.bias"}:
        raise ValueError("Expected an archived identity/mean/scalar linear head")
    weight, bias = state["head.linear.weight"], state["head.linear.bias"]
    if (
        weight.shape != (1, FEATURES)
        or bias.shape != (1,)
        or not torch.isfinite(weight).all()
        or not torch.isfinite(bias).all()
    ):
        raise ValueError("Invalid archived scalar linear head tensors")
    provenance = {
        "seed": seed,
        "checkpoint_sha256": file_digest(checkpoint_path),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "checkpoint_logged_step": selected["logged_step"],
    }
    return provenance, weight[0].double().numpy(), float(bias.item())


def _local_head(spec: Path, clean_uri: str) -> tuple[dict[str, Any], np.ndarray, float]:
    paths = json.loads(spec.read_text())
    raw_history = json.loads((spec.parent / paths["metric_history"]).read_text())
    history = {
        key: [
            SimpleNamespace(step=row["step"], value=row["value"])
            for row in raw_history[key]
        ]
        for key in METRICS
    }
    if any(
        type(item.step) is not int or item.step < 0 or not math.isfinite(item.value)
        for series in history.values()
        for item in series
    ):
        raise ValueError(
            "Local metric history must contain finite values at integer steps"
        )
    return _head(
        (spec.parent / paths["config"]).read_text(),
        spec.parent / paths["checkpoint"],
        history,
        clean_uri,
    )


def _mlflow_head(
    run_id: str, clean_uri: str
) -> tuple[dict[str, Any], np.ndarray, float]:
    from mlflow import MlflowClient
    from mlflow.artifacts import download_artifacts, load_text

    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("--run-id requires an explicit 32-character MLflow run ID")
    client = MlflowClient()
    run = client.get_run(run_id)
    revision = run.data.tags.get("mlflow.source.git.commit")
    if run.info.status != "FINISHED" or not revision:
        raise ValueError(
            "Expected a finished archived run with a recorded source commit"
        )
    config = load_text(f"runs:/{run_id}/{archive.CONFIG}")
    history = {key: client.get_metric_history(run_id, key) for key in METRICS}
    with tempfile.TemporaryDirectory(prefix="l2-rotation-checkpoint-") as scratch:
        path = Path(
            download_artifacts(
                artifact_uri=f"runs:/{run_id}/{archive.CHECKPOINT}", dst_path=scratch
            )
        )
        provenance, weight, bias = _head(config, path, history, clean_uri)
    provenance.update(run_id=run_id, code_revision=revision)
    return provenance, weight, bias


def _vector_batches(path: Path, coords: pa.Table) -> Iterator[np.ndarray]:
    if not _coordinates(path).equals(coords):
        raise ValueError(f"Ordered tile coordinates differ for {path.name}")
    parquet = pq.ParquetFile(path)
    vector_type = parquet.schema_arrow.field("embedding").type
    fixed = pa.types.is_fixed_size_list(vector_type)
    if (
        not (
            fixed
            or pa.types.is_list(vector_type)
            or pa.types.is_large_list(vector_type)
        )
        or vector_type.value_type != pa.float32()
        or (fixed and vector_type.list_size != FEATURES)
    ):
        raise ValueError(f"Invalid FP32 embedding schema in {path.name}")
    for batch in parquet.iter_batches(columns=["embedding"], batch_size=4096):
        vectors = batch[0]
        if vectors.null_count or (
            not fixed and not np.all(vectors.value_lengths().to_numpy() == FEATURES)
        ):
            raise ValueError(f"Invalid embedding widths or null rows in {path.name}")
        values = vectors.flatten()
        if values.null_count:
            raise ValueError(f"Null embedding elements in {path.name}")
        array = values.to_numpy().reshape(-1, FEATURES)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite embedding elements in {path.name}")
        yield array


def _mean(path: Path, coords: pa.Table) -> np.ndarray:
    total = np.zeros(FEATURES)
    for array in _vector_batches(path, coords):
        total += array.sum(axis=0, dtype=np.float64)
    return total / coords.num_rows


def _paired_clean(
    archived_path: Path, fresh_path: Path, coords: pa.Table
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archived_total = np.zeros(FEATURES)
    fresh_total = np.zeros(FEATURES)
    relative = []
    for archived, fresh in zip(
        _vector_batches(archived_path, coords),
        _vector_batches(fresh_path, coords),
        strict=True,
    ):
        if archived.shape != fresh.shape:
            raise ValueError(f"Unmatched clean embedding rows for {archived_path.name}")
        archived_total += archived.sum(axis=0, dtype=np.float64)
        fresh_total += fresh.sum(axis=0, dtype=np.float64)
        squared_archived = np.einsum("ij,ij->i", archived, archived, dtype=np.float64)
        if (squared_archived == 0).any():
            raise ValueError(
                f"Undefined relative L2 for zero vectors in {archived_path.name}"
            )
        difference = np.subtract(fresh, archived, dtype=np.float64)
        squared_difference = np.einsum("ij,ij->i", difference, difference)
        relative.append(np.sqrt(squared_difference / squared_archived))
    return (
        archived_total / coords.num_rows,
        fresh_total / coords.num_rows,
        np.concatenate(relative),
    )


def _change(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "signed_mean": float(values.mean()),
        "mean_absolute": float(absolute.mean()),
        "median_absolute": float(np.median(absolute)),
        "max_absolute": float(absolute.max()),
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    if manifest["split"] != "val" or manifest["subset"] is not True:
        raise ValueError("Expected an explicit frozen L2 validation subset")
    if file_digest(args.data_mapping) != manifest["mapping_sha256"]:
        raise ValueError("The data mapping differs from the frozen probe")
    mapping = archive.mapping_rows(args.data_mapping)
    record_splits: dict[str, set[str]] = {}
    for split, record, _ in mapping.values():
        record_splits.setdefault(record, set()).add(split)
    if any(len(splits) != 1 for splits in record_splits.values()):
        raise ValueError("The data mapping contains a record in multiple splits")
    clean = _fresh_config(args.clean_output, manifest, "clean")
    rotated = _fresh_config(args.rotation_output, manifest, "rotation")
    if {key: value for key, value in clean.items() if key != "tile_view"} != {
        key: value for key, value in rotated.items() if key != "tile_view"
    }:
        raise ValueError("The fresh views have different encoder or runtime provenance")

    clean_uri = manifest["clean_uri"]
    heads = [_local_head(path, clean_uri) for path in args.local_spec]
    heads.extend(_mlflow_head(run_id, clean_uri) for run_id in args.run_id)
    if not heads or len({record["seed"] for record, _, _ in heads}) != len(heads):
        raise ValueError(
            "Provide at least one baseline and only one checkpoint per seed"
        )

    archived = _L2Source(clean_uri)
    views = ("archived", "fresh_clean", "fresh_rot90")
    means: dict[str, list[np.ndarray]] = {view: [] for view in views}
    relative_l2 = []
    targets = []
    for slide in manifest["slides"]:
        stem = slide["slide_id"]
        mapped_split, _, target = mapping.get(stem, (None, None, None))
        if mapped_split != "val" or target is None or stem not in archived.files:
            raise ValueError(
                f"The frozen slide is outside the eligible validation cohort: {stem}"
            )
        if not all(
            is_complete(root, slide)
            for root in (args.clean_output, args.rotation_output)
        ):
            raise ValueError(f"Missing or invalid fresh embedding receipt for {stem}")
        table = _selection(args.manifest.parent, slide)
        if selection_digest(table) != slide["selection_digest"]:
            raise ValueError(f"The frozen coordinates changed for {stem}")
        coords = table.select(["x", "y"])
        name = f"{stem}.parquet"
        with archived.opened(stem) as path:
            archived_mean, fresh_mean, tile_errors = _paired_clean(
                path, args.clean_output / "embeddings" / name, coords
            )
        means["archived"].append(archived_mean)
        means["fresh_clean"].append(fresh_mean)
        means["fresh_rot90"].append(
            _mean(args.rotation_output / "embeddings" / name, coords)
        )
        relative_l2.append(tile_errors)
        targets.append(target)

    p95, p99 = map(float, np.quantile(np.concatenate(relative_l2), [0.95, 0.99]))
    embedding_parity = p99 <= args.parity_max_p99_relative_l2
    matrix = {view: np.stack(vectors) for view, vectors in means.items()}
    target_values = np.asarray(targets)
    seeds = []
    for provenance, weight, bias in sorted(heads, key=lambda head: head[0]["seed"]):
        pred = {view: matrix[view] @ weight + bias for view in views}
        mae = {view: float(np.abs(pred[view] - target_values).mean()) for view in views}
        drift = _change(pred["fresh_clean"] - pred["archived"])
        shift = _change(pred["fresh_rot90"] - pred["fresh_clean"])
        parity = drift["max_absolute"] <= args.parity_max_abs_drift
        seeds.append(
            {
                **provenance,
                "validation_subset_mae": mae,
                "rotation_mae_change": mae["fresh_rot90"] - mae["fresh_clean"],
                "archived_to_fresh_clean_prediction_drift": drift,
                "fresh_clean_to_rot90_prediction_shift": shift,
                "clean_prediction_parity": parity,
                "rotation_material": (
                    embedding_parity
                    and parity
                    and shift["mean_absolute"] >= args.rotation_min_mean_abs_shift
                    and shift["mean_absolute"] > drift["mean_absolute"]
                ),
            }
        )
    prediction_parity = all(seed["clean_prediction_parity"] for seed in seeds)
    return {
        "manifest_digest": manifest["digest"],
        "mapping_sha256": manifest["mapping_sha256"],
        "archived_clean_uri": clean_uri,
        "slides": len(targets),
        "tiles": sum(slide["tile_count"] for slide in manifest["slides"]),
        "fresh_runs": {"clean": clean, "rotation": rotated},
        "rotation_degrees_ccw": 90,
        "clean_embedding_relative_l2": {
            "aggregation": "pooled across slides, equally weighted by tile",
            "p95": p95,
            "p99": p99,
        },
        "thresholds": {
            "parity_max_abs_drift": args.parity_max_abs_drift,
            "parity_max_p99_relative_l2": args.parity_max_p99_relative_l2,
            "rotation_min_mean_abs_shift": args.rotation_min_mean_abs_shift,
        },
        "clean_embedding_parity": embedding_parity,
        "clean_prediction_parity_all_seeds": prediction_parity,
        "clean_parity_all_seeds": embedding_parity and prediction_parity,
        "rotation_material_all_seeds": all(seed["rotation_material"] for seed in seeds),
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The fresh paths are production directories containing run.json, "
            "embeddings/ and complete/. Local metric history maps valid/loss, "
            "valid/mae and valid/thresholded_f1 to lists of step/value objects. "
            "Prespecify prediction thresholds in MammaPrint index units and "
            "tile parity as unitless relative L2 before scoring."
        ),
    )
    for flag in ("manifest", "clean-output", "rotation-output", "data-mapping"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Authorized archived MLflow run ID",
    )
    parser.add_argument(
        "--local-spec",
        action="append",
        default=[],
        type=Path,
        help="JSON with checkpoint, config and metric_history paths relative to the spec",
    )
    parser.add_argument("--parity-max-abs-drift", type=float, required=True)
    parser.add_argument(
        "--parity-max-p99-relative-l2",
        type=float,
        required=True,
        help="Maximum pooled per-tile p99 of norm(fresh - archived) / norm(archived)",
    )
    parser.add_argument("--rotation-min-mean-abs-shift", type=float, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New aggregate JSON report"
    )
    args = parser.parse_args()
    if not (args.run_id or args.local_spec):
        parser.error("Provide an explicit archived --run-id or --local-spec")
    if (
        not math.isfinite(args.parity_max_abs_drift)
        or args.parity_max_abs_drift < 0
        or not math.isfinite(args.parity_max_p99_relative_l2)
        or args.parity_max_p99_relative_l2 < 0
        or not math.isfinite(args.rotation_min_mean_abs_shift)
        or args.rotation_min_mean_abs_shift <= 0
    ):
        parser.error(
            "Parity thresholds must be finite and nonnegative. The rotation threshold must be finite and positive."
        )
    if args.output.exists():
        parser.error("Output report already exists")
    result = json.dumps(score(args), indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as output:
        output.write(result)


if __name__ == "__main__":
    main()
