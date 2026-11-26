"""Render or execute the twelve validation-only L1 and L2 training fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
import uuid
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict


if TYPE_CHECKING:
    from collections.abc import Mapping

    from mlflow import MlflowClient
    from omegaconf import DictConfig


ROOT = Path(__file__).resolve().parents[2]
LEVELS = (1, 2)
POOLINGS = ("mean", "attention")
SEEDS = (0, 1, 2)


class TrainingManifest(TypedDict):
    digest: str
    data_mapping_digest: str
    splits: dict[str, dict[str, int]]


class FitKey(TypedDict):
    level: int
    pooling: str
    seed: int


class FitRecord(TypedDict):
    fit: FitKey
    mlflow_run_id: str
    bundle_digest: str
    checkpoint: str
    checkpoint_digest: str
    config: str
    config_digest: str


type MappingRow = dict[str, str]


class Fit(NamedTuple):
    level: int
    pooling: str
    seed: int

    @property
    def name(self) -> str:
        return f"resolution-l{self.level}-{self.pooling}-s{self.seed}"

    def key(self) -> FitKey:
        return {"level": self.level, "pooling": self.pooling, "seed": self.seed}


FITS = tuple(Fit(*values) for values in product(LEVELS, POOLINGS, SEEDS))


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(bundle: Path, data_mapping: Path) -> TrainingManifest:
    manifest: TrainingManifest = json.loads((bundle / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "digest"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if manifest.get("digest") != digest:
        raise ValueError("Comparison manifest has an invalid digest")
    if manifest.get("data_mapping_digest") != file_digest(data_mapping):
        raise ValueError("The split mapping no longer matches the comparison manifest")
    return manifest


def split_rows(
    data_mapping: Path, manifest: TrainingManifest, split: str
) -> dict[str, MappingRow]:
    expected = manifest["splits"][split]
    if not expected or any(
        not isinstance(count, int) or not 1 <= count <= 3000
        for count in expected.values()
    ):
        raise ValueError(f"{split} must contain nonempty bags of at most 3,000 tiles")
    rows: dict[str, MappingRow] = {}
    with data_mapping.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != split or not row["path"]:
                continue
            slide_id = Path(row["path"]).stem
            if slide_id not in expected:
                continue
            try:
                labeled = math.isfinite(float(row["mammaprint_index"]))
            except (TypeError, ValueError):
                labeled = False
            if slide_id in rows or not labeled:
                raise ValueError(
                    f"Duplicate slide or missing label in {split}: {slide_id}"
                )
            rows[slide_id] = row
    if set(rows) != set(expected):
        raise ValueError(
            f"Mapping is missing {split} slides: {sorted(expected.keys() - rows.keys())}"
        )
    return rows


def verify_split(
    bundle: Path,
    level: int,
    split: str,
    expected: dict[str, int],
) -> Path:
    import pyarrow.parquet as pq

    directory = bundle / f"l{level}" / split
    files = {path.stem: path for path in directory.glob("*.parquet")}
    if files.keys() != expected.keys():
        raise ValueError(
            f"{directory} has unexpected or missing slides: {sorted(files.keys() ^ expected.keys())}"
        )
    for slide, count in expected.items():
        if pq.read_metadata(files[slide]).num_rows != count:
            raise ValueError(f"{directory} has a mismatched bag count for {slide}")
    return directory


def write_json(path: Path, content: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(content, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def fit_overrides(fit: Fit, bundle: Path, attempt: Path, digest: str) -> list[str]:
    dirs = ",".join(
        f"{split}:{json.dumps(str(bundle / f'l{fit.level}' / split))}"
        for split in ("train", "val")
    )
    return [
        "+experiment=ml/train_mil_embeddings_regression",
        "~data/embedded",
        "~datamodule.test",
        f"dataset.levels={{{fit.level}:{{uris:{{{dirs}}}}}}}",
        f"dataset.data_mapping={json.dumps(str(attempt / 'fit_mapping.csv'))}",
        f"ml/aggregator={fit.pooling}",
        f"seed={fit.seed}",
        "mode=fit",
        "test_after_fit=false",
        "+ml.metrics.spearman._target_=torchmetrics.SpearmanCorrCoef",
        "+ml.metrics.sign_auroc._target_=ml.metrics.SignAUROC",
        "+ml.metrics.sign_auroc.metric._target_=torchmetrics.classification.BinaryAUROC",
        f"metadata.run_name={fit.name}",
        f"metadata.description={json.dumps('Matched L1/L2 validation-only MammaPrint regression')}",
        f"+logger.tags.resolution_bundle={digest}",
        "trainer.devices=1",
        "datamodule.num_workers=2",
        "datamodule.pin_memory=true",
        "datamodule.persistent_workers=true",
        f"++trainer.callbacks.model_checkpoint.dirpath={json.dumps(str(attempt))}",
        "++trainer.callbacks.model_checkpoint.enable_version_counter=false",
    ]


def _write_fit_mapping(path: Path, rows: list[MappingRow]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compose_fit(overrides: list[str]) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import ml.train  # noqa: F401  # Registers the shared seed resolver.

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="ml", overrides=overrides)
    if config.mode != "fit" or config.test_after_fit or "test" in config.datamodule:
        raise ValueError(
            "Resolution comparison fits must build only train and validation loaders"
        )
    OmegaConf.resolve(config)
    return config


def _create_run(
    run_name: str, bundle_digest: str, fit: Fit
) -> tuple[MlflowClient, str]:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    experiment_name = "MammaPrint"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            experiment_id = client.create_experiment(experiment_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise
            experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id
    tags = {
        "resolution_bundle": bundle_digest,
        "resolution_level": str(fit.level),
        "resolution_pooling": fit.pooling,
        "resolution_seed": str(fit.seed),
        "mlflow.source.git.commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    return client, client.create_run(
        experiment_id, tags=tags, run_name=run_name
    ).info.run_id


def _execute_fit(
    fit: Fit,
    args: argparse.Namespace,
    manifest: TrainingManifest,
    rows: list[MappingRow],
) -> None:
    import fcntl

    from omegaconf import OmegaConf

    directory = args.output / fit.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata_path = directory / "fit.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata["bundle_digest"] == manifest["digest"]
                and file_digest(directory / metadata["checkpoint"])
                == metadata["checkpoint_digest"]
                and file_digest(directory / metadata["config"])
                == metadata["config_digest"]
            ):
                print(
                    f"Completed fit already exists for {fit.name} ({metadata['mlflow_run_id']})"
                )
                return
            raise ValueError(
                f"Existing fit provenance differs for {fit.name}. Use a new output root."
            )
        attempt = directory / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        _write_fit_mapping(attempt / "fit_mapping.csv", rows)
        overrides = fit_overrides(fit, args.bundle, attempt, manifest["digest"])
        config = _compose_fit(overrides)
        config_path = attempt / "config.yaml"
        OmegaConf.save(config, config_path)
        client, run_id = _create_run(fit.name, manifest["digest"], fit)
        command = [
            sys.executable,
            "-m",
            "ml.train",
            *overrides,
            f"+logger.run_id={run_id}",
        ]
        checkpoint = attempt / "best.ckpt"
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
            record: FitRecord = {
                "fit": fit.key(),
                "mlflow_run_id": run_id,
                "bundle_digest": manifest["digest"],
                "checkpoint": str(checkpoint.relative_to(directory)),
                "checkpoint_digest": file_digest(checkpoint),
                "config": str(config_path.relative_to(directory)),
                "config_digest": file_digest(config_path),
            }
            write_json(metadata_path, record)
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=LEVELS)
    parser.add_argument("--pooling", choices=POOLINGS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument(
        "--execute", action="store_true", help="Run fits in the current environment"
    )
    args = parser.parse_args()
    for key in ("bundle", "data_mapping", "output"):
        setattr(args, key, getattr(args, key).resolve())
    fits = [
        fit
        for fit in FITS
        if (args.level is None or fit.level == args.level)
        and (args.pooling is None or fit.pooling == args.pooling)
        and (args.seed is None or fit.seed == args.seed)
    ]
    if not args.execute:
        for fit in fits:
            attempt = args.output / fit.name / "attempts" / "ATTEMPT_ID"
            command = [sys.executable, "-m", "ml.train"]
            command.extend(fit_overrides(fit, args.bundle, attempt, "BUNDLE_DIGEST"))
            command.append("+logger.run_id=MLFLOW_RUN_ID")
            print(shlex.join(command))
        return

    manifest = load_manifest(args.bundle, args.data_mapping)
    rows: list[MappingRow] = []
    for split in ("train", "val"):
        for level in sorted({fit.level for fit in fits}):
            verify_split(args.bundle, level, split, manifest["splits"][split])
        rows.extend(split_rows(args.data_mapping, manifest, split).values())
    for fit in fits:
        _execute_fit(fit, args, manifest, rows)


if __name__ == "__main__":
    main()
