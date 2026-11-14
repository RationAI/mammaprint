"""Read-only source-run recovery and native frozen-model inference."""

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from ml.models.module import MammaprintModule


REPOSITORY = Path(__file__).resolve().parents[2]
RESOLVED_CONFIG = "configs/config-resolved.yaml"
BEST_CHECKPOINT = "checkpoints/best/checkpoint.ckpt"


@dataclass
class SourceRun:
    run_id: str
    training_sha: str | None
    config: DictConfig
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    datasets: dict[str, Any] = field(default_factory=dict, repr=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_training_revision(training_sha: str | None) -> None:
    """Require a known revision with identical inference/data implementation.

    Fetch missing historical Git objects explicitly before invoking this tool;
    this check never downloads code or guesses from a mutable branch name.
    """
    if training_sha is None or re.fullmatch(r"[0-9a-fA-F]{40}", training_sha) is None:
        raise ValueError("An exact 40-character training Git revision is required.")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{training_sha}^{{commit}}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                training_sha,
                "--",
                "ml/models",
                "ml/data",
                "ml/typing.py",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Training revision {training_sha} is missing locally or inference/data code differs. "
            "Fetch the revision and resolve compatibility before explaining this checkpoint."
        ) from exc


def load_source_run(
    run_id: str,
    cache_dir: Path,
    *,
    download_weights: bool = True,
    client: Any = None,
    training_sha: str | None = None,
) -> SourceRun:
    """Download only the original resolved config and (optionally) checkpoint.

    `training_sha` is an explicit provenance recovery override for older runs
    lacking the standard MLflow tag. A conflicting tag is never overridden.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("source_run_id must be a safe MLflow run identifier.")
    if client is None:
        from mlflow import MlflowClient

        client = MlflowClient()
    run = client.get_run(run_id)
    recorded = run.data.tags.get("mlflow.source.git.commit")
    if recorded and training_sha and recorded != training_sha:
        raise ValueError(
            "Explicit training revision conflicts with the source run's Git tag."
        )
    revision = recorded or training_sha
    verify_training_revision(revision)
    destination = cache_dir / run_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        config_path = Path(
            client.download_artifacts(run_id, RESOLVED_CONFIG, str(destination))
        )
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig) or "ml" not in config:
            raise ValueError("No model graph in source config")
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    except Exception as exc:
        raise RuntimeError(
            f"Run {run_id}: cannot read a complete resolved config ({RESOLVED_CONFIG})."
        ) from exc
    checkpoint = None
    checksum = None
    if download_weights:
        checkpoint = Path(
            client.download_artifacts(run_id, BEST_CHECKPOINT, str(destination))
        )
        checksum = file_sha256(checkpoint)
    return SourceRun(run_id, revision, config, checkpoint, checksum)


def load_frozen_model(source: SourceRun) -> MammaprintModule:
    if source.checkpoint_path is None:
        raise ValueError("A checkpoint is required to evaluate model effects.")
    module = instantiate(source.config.ml)
    if not isinstance(module, MammaprintModule):
        raise TypeError("Only the original MammaprintModule is supported.")
    if not isinstance(module.output_activation, torch.nn.Identity):
        raise ValueError(
            "Only Identity output_activation is supported by the raw-logit/index adapter."
        )
    # Own trusted Lightning checkpoints include non-tensor callback state.
    checkpoint = torch.load(
        source.checkpoint_path, map_location="cpu", weights_only=False
    )
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval().requires_grad_(False)
    return module


@torch.inference_mode()
def model_outputs(module: MammaprintModule, bag: Tensor, label_mode: str) -> Tensor:
    """Native raw outputs, with no IBD analytic readout or joint softmax."""
    expected = 2 if label_mode == "both" else 1
    if label_mode not in {"type", "index", "both"}:
        raise ValueError(f"Unknown label mode: {label_mode}")
    module.eval()
    pooled, _ = module.aggregator(module.encoder(bag))
    outputs = module.head(pooled).reshape(-1)
    if outputs.numel() != expected or not bool(torch.isfinite(outputs).all()):
        raise ValueError("Source model emitted invalid outputs for its label mode.")
    return outputs
