"""Reconcile L1 shards, publish MLflow splits, and create equal-count MIL bags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from preprocessing.embedding_selection import selection_digest
from preprocessing.resolution_embeddings import (
    FEATURES,
    file_digest,
    is_complete,
    load_selection_manifest,
    lock,
    write_json,
)


SPLITS = ("train", "val", "test")


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mapping(path: Path) -> dict[str, tuple[str, bool]]:
    mapping = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in SPLITS or not row["path"]:
                continue
            stem = Path(row["path"]).stem
            if stem in mapping:
                raise ValueError(f"Duplicate mapping slide {stem}")
            try:
                labeled = np.isfinite(float(row["mammaprint_index"]))
            except (TypeError, ValueError):
                labeled = False
            mapping[stem] = (row["split"], bool(labeled))
    return mapping


def _link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _features(path: Path, slide: dict[str, Any], cap: int) -> pa.Table:
    table = pq.read_table(path)
    if (
        table.num_rows != slide["tile_count"]
        or selection_digest(table) != slide["selection_digest"]
    ):
        raise ValueError(f"Embedding coordinates, scores, or ranks changed for {path}")
    if not table.num_rows or cap > table.num_rows:
        raise ValueError(f"Insufficient embedding rows for {path}")
    features = table["embedding"].combine_chunks()
    if features.type != pa.list_(pa.float32(), FEATURES) or features.null_count:
        raise ValueError(f"Invalid feature schema for {path}")
    if "l2_embedding_digest" in slide:
        encoded = np.asarray(features.values.to_numpy(), dtype="<f4").tobytes()
        if hashlib.sha256(encoded).hexdigest() != slide["l2_embedding_digest"]:
            raise ValueError(f"L2 features changed for {path}")
    if cap != table.num_rows:
        table = table.filter(pc.less(table["sample_rank"], cap))
    if table.num_rows != cap:
        raise ValueError(f"Unexpected rank distribution for {path}")
    return table


def _attempts(
    production: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str], float]:
    failures: dict[str, list[dict[str, str]]] = {}
    reports = []
    wall_seconds = 0.0
    for report in sorted((production / "attempts").glob("*/*.json")):
        attempt = _load(report)
        reports.append(str(report.relative_to(production)))
        for slide, problem in attempt.get("failures", {}).items():
            failures.setdefault(slide, []).append(
                {"report": reports[-1], "reason": problem}
            )
        if attempt.get("device") != "cpu":
            wall_seconds += attempt.get("wall_seconds", 0.0)
    return failures, reports, wall_seconds / 3600


type SlideTable = dict[str, dict[str, Any]]
type RegistrationInputs = tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, tuple[str, bool]],
    SlideTable,
    SlideTable,
    dict[str, Any],
]


def _inspect(args: argparse.Namespace) -> RegistrationInputs:
    l1 = load_selection_manifest(args.l1_manifest)
    l2 = load_selection_manifest(args.l2_manifest)
    production = args.l1_output / "production"
    provenance = _load(production / "run.json")
    mapping_digest = file_digest(args.data_mapping)
    if (l1["level"], l2["level"]) != (1, 2):
        raise ValueError("Expected L1 then L2 selection manifests")
    if any(manifest["data_mapping_sha256"] != mapping_digest for manifest in (l1, l2)):
        raise ValueError("Selection manifests require the same frozen mapping CSV")
    if (l1["cap"], l1["seed"]) != (l2["cap"], l2["seed"]):
        raise ValueError("Selection manifests require matching cap and random seed")
    if provenance["manifest_digest"] != l1["digest"]:
        raise ValueError("L1 worker output belongs to a different manifest")
    mapping = _mapping(args.data_mapping)

    def verify_slides(manifest: dict[str, Any]) -> SlideTable:
        slides = {slide["slide_id"]: slide for slide in manifest["slides"]}
        if len(slides) != len(manifest["slides"]):
            raise ValueError("Duplicate manifest slide IDs")
        for stem, slide in slides.items():
            if (
                Path(stem).name != stem
                or stem not in mapping
                or slide["split"] != mapping[stem][0]
            ):
                raise ValueError(f"Invalid manifest identity or split for {stem}")
        return slides

    slides1, slides2 = verify_slides(l1), verify_slides(l2)
    attempts, reports, gpu_hours = _attempts(production)
    inventory: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    failures = []
    hard_failures = False
    for stem, slide in slides1.items():
        split = slide["split"]
        if not slide["tile_count"]:
            failures.append(
                {"slide_id": stem, "split": split, "reason": "zero_eligible_tiles"}
            )
        elif is_complete(production, slide):
            path = production / "complete" / f"{stem}.json"
            inventory[split][stem] = _load(path)["sha256"]
        else:
            failures.append(
                {
                    "slide_id": stem,
                    "split": split,
                    "reason": "missing_or_invalid_embedding",
                    "attempts": attempts.get(stem, []),
                }
            )
            hard_failures = True
    for stem, (split, _) in mapping.items():
        if stem not in slides1:
            failures.append(
                {"slide_id": stem, "split": split, "reason": "absent_from_l1_tiling"}
            )
    registration = {
        "source_manifest_digest": l1["digest"],
        "data_mapping_digest": mapping_digest,
        "embedding_config": provenance,
        "cap": l1["cap"],
        "sampling_seed": l1["seed"],
        "physical_scale": {
            "nominal_mpp": 0.485,
            "source_tile_px": 224,
            "source_stride_px": 112,
        },
        "grid": "fixed phase every second source coordinate per axis",
        "feature_dim": FEATURES,
        "successful": inventory,
        "split_counts": {
            split: {
                "slides": len(inventory[split]),
                "tiles": sum(slides1[stem]["tile_count"] for stem in inventory[split]),
            }
            for split in SPLITS
        },
        "failures": failures,
        "worker_reports": reports,
        "worker_wall_gpu_hours": gpu_hours,
        "source_selection": l1,
    }
    write_json(args.output / "l1_inventory.json", registration)
    if hard_failures and not args.allow_incomplete:
        raise ValueError(
            "Missing or invalid L1 output. See l1_inventory.json, retry affected shards, "
            "or use --allow-incomplete after reviewing the exceptions."
        )
    return l1, l2, production, mapping, slides1, slides2, registration


def _make_bundle(
    args: argparse.Namespace,
    l1: dict[str, Any],
    l2: dict[str, Any],
    production: Path,
    mapping: dict[str, tuple[str, bool]],
    slides1: SlideTable,
    slides2: SlideTable,
    registration: dict[str, Any],
) -> Path:
    output = args.output / "comparison"
    matched: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    excluded: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    successes = registration["successful"]
    for stem, (split, labeled) in mapping.items():
        l2_slide = slides2.get(stem)
        if stem not in successes[split]:
            excluded[split][stem] = "no_valid_l1_embedding"
        elif not l2_slide or not l2_slide["tile_count"]:
            excluded[split][stem] = (
                l2_slide["status"] if l2_slide else "absent_from_l2_tiling"
            )
        elif not labeled:
            excluded[split][stem] = "missing_index_label"
        else:
            matched[split][stem] = min(
                slides1[stem]["tile_count"], l2_slide["tile_count"]
            )
    if any(not slides for slides in matched.values()):
        raise ValueError("At least one comparison split has no shared labeled slides")
    unmatched_l2 = [
        stem
        for split in SPLITS
        for stem, reason in excluded[split].items()
        if reason
        in {
            "absent_from_l2_tiling",
            "l2_embedding_missing",
            "l2_coordinate_mismatch",
            "l2_invalid_embeddings",
        }
    ]
    if unmatched_l2 and not args.allow_incomplete:
        raise ValueError(
            f"{len(unmatched_l2)} L2 controls are unresolved; examples {unmatched_l2[:10]}"
        )
    realized_scale = {}
    for split, counts in matched.items():
        ratios = np.array(
            [
                [
                    slides2[stem][axis] / slides1[stem][axis]
                    for axis in ("mpp_x", "mpp_y")
                ]
                for stem in counts
            ]
        )
        realized_scale[split] = {
            "same_pyramid_level": sum(
                slides1[stem]["level"] == slides2[stem]["level"] for stem in counts
            ),
            "l1_not_finer_on_both_axes": int(
                np.count_nonzero((ratios <= 1).any(axis=1))
            ),
            "l2_over_l1_mpp_x_min_median_max": np.quantile(
                ratios[:, 0], [0, 0.5, 1]
            ).tolist(),
            "l2_over_l1_mpp_y_min_median_max": np.quantile(
                ratios[:, 1], [0, 0.5, 1]
            ).tolist(),
        }
    manifest = {
        "data_mapping_digest": registration["data_mapping_digest"],
        "source_manifests": {"1": l1["digest"], "2": l2["digest"]},
        "l1_embedding_config": registration["embedding_config"],
        "l1_embeddings": {
            split: {stem: successes[split][stem] for stem in matched[split]}
            for split in SPLITS
        },
        "splits": matched,
        "realized_scale": realized_scale,
        "exclusions": excluded,
    }
    if output.exists():
        existing = _load(output / "manifest.json")
        digest = existing.pop("digest")
        if (
            digest != _digest(existing)
            or {key: value for key, value in existing.items() if key != "files"}
            != manifest
        ):
            raise ValueError(
                f"Bundle directory already pins a different manifest: {output}"
            )
        for level in (1, 2):
            for split in SPLITS:
                files = {
                    path.stem: pq.read_metadata(path).num_rows
                    for path in (output / f"l{level}" / split).glob("*.parquet")
                }
                if files != matched[split]:
                    raise ValueError(f"Existing bundle is invalid at l{level}/{split}")
                for stem, expected in existing["files"][f"l{level}"][split].items():
                    if (
                        file_digest(output / f"l{level}" / split / f"{stem}.parquet")
                        != expected
                    ):
                        raise ValueError(
                            f"Comparison file changed at l{level}/{split}/{stem}.parquet"
                        )
        return output

    sources = (production / "embeddings", args.l2_manifest.parent / "embeddings")
    staging = Path(tempfile.mkdtemp(prefix=".comparison-", dir=args.output))
    manifest["files"] = {
        f"l{level}": {split: {} for split in SPLITS} for level in (1, 2)
    }
    try:
        for level, directory in enumerate(sources, start=1):
            slides = slides1 if level == 1 else slides2
            for split, counts in matched.items():
                dest = staging / f"l{level}" / split
                dest.mkdir(parents=True)
                for stem, count in sorted(counts.items()):
                    source = directory / f"{stem}.parquet"
                    target = dest / source.name
                    table = _features(source, slides[stem], count)
                    if count == slides[stem]["tile_count"]:
                        _link(source, target)
                    else:
                        pq.write_table(table, target)
                    manifest["files"][f"l{level}"][split][stem] = file_digest(target)
        manifest["digest"] = _digest(manifest)
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
    return output


def _publish(
    args: argparse.Namespace, production: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"MLflow experiment {args.experiment!r} does not exist")
    inventory = registration["successful"]
    state_path = args.output / "mlflow_publication.json"
    tracking_uri = client.tracking_uri
    fingerprint = _digest(
        {
            "inventory": inventory,
            "embedding_config": registration["embedding_config"],
            "data_mapping_digest": registration["data_mapping_digest"],
        }
    )
    if state_path.exists():
        saved = _load(state_path)
        if (saved["fingerprint"], saved["tracking_uri"]) != (fingerprint, tracking_uri):
            raise ValueError(
                "Publication directory belongs to another dataset or tracking server"
            )
        run_id = saved["run_id"]
    else:
        run = client.create_run(
            experiment.experiment_id, run_name="L1 cancer mask Virchow2 capped"
        )
        run_id = run.info.run_id
        write_json(
            state_path,
            {
                "run_id": run_id,
                "fingerprint": fingerprint,
                "tracking_uri": tracking_uri,
            },
        )
    client.log_artifact(
        run_id, str(args.output / "l1_inventory.json"), artifact_path="metadata"
    )
    all_files = {
        stem: digest
        for entries in inventory.values()
        for stem, digest in entries.items()
    }
    for part, files in {"all": all_files, **inventory}.items():
        remote_path = f"embeddings_{part}"
        local = production / "embeddings"
        expected = {
            f"{stem}.parquet": (local / f"{stem}.parquet").stat().st_size
            for stem in files
        }

        def list_remote(path: str) -> dict[str, int | None]:
            return {
                Path(item.path).name: item.file_size
                for item in client.list_artifacts(run_id, path)
                if not item.is_dir
            }

        remote = list_remote(remote_path)
        extra = set(remote) - set(expected)
        if extra:
            raise ValueError(
                f"Unexpected MLflow files in {remote_path}: {sorted(extra)[:10]}"
            )
        for filename, size in sorted(expected.items()):
            if remote.get(filename) != size:
                client.log_artifact(
                    run_id, str(local / filename), artifact_path=remote_path
                )
        if list_remote(remote_path) != expected:
            raise ValueError(f"MLflow file inventory mismatch for {remote_path}")
    client.set_tag(
        run_id, "resolution_manifest", registration["source_manifest_digest"]
    )
    root = client.get_run(run_id).info.artifact_uri.rstrip("/")
    uris = {part: f"{root}/embeddings_{part}" for part in ("all", *SPLITS)}
    card = {
        "name": "l1_cancer_mask_5_embed",
        "level": 1,
        "mpp": 0.485,
        "tile_extent": 224,
        "stride": 224,
        "source_stride": 112,
        "uris": uris,
        "provenance": f"{root}/metadata/l1_inventory.json",
    }
    write_json(args.output / "l1_data_card.json", card)
    client.log_artifact(
        run_id, str(args.output / "l1_data_card.json"), artifact_path="metadata"
    )
    client.set_terminated(run_id)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--l2-manifest", type=Path, required=True)
    parser.add_argument("--l1-output", type=Path, required=True)
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish", action="store_true", help="Write verified splits to MLflow"
    )
    parser.add_argument("--experiment", default="MammaPrint")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    with ExitStack() as locks:
        locks.enter_context(lock(args.output / ".reconcile.lock"))
        for shard in sorted(
            {
                slide["shard"]
                for slide in load_selection_manifest(args.l1_manifest)["slides"]
            }
        ):
            locks.enter_context(
                lock(args.l1_output / "production" / "locks" / f"shard-{shard}.lock")
            )
        info = _inspect(args)
        bundle = _make_bundle(args, *info)
        result: dict[str, Any] = {
            "bundle": str(bundle),
            "l1_inventory": str(args.output / "l1_inventory.json"),
        }
        if args.publish:
            result["l1_card"] = _publish(args, info[2], info[-1])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
