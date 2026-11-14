"""Read-only source-run recovery and native frozen-model inference."""

import ast
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


# Reviewed transport implementations: original direct download and resumable
# download. These are AST fingerprints of ONLY the imports/functions below, not
# training revisions. New transfer behavior requires a fresh compatibility review.
_REVIEWED_TRANSPORT = {
    "f9c2a0338570509d5636548e8cc654fbc29f18266531fa80d023d7f0fc010ecb",
    "7869826394396fa6b23669152c83136713a6e05c1e6e45eb00e049231e4c73a4",
}
_TRANSPORT_FUNCTIONS = {
    "download_artifacts_with_retries",
    "_artifact_files",
    "_is_complete_download",
    "_download_incomplete_artifacts",
}
_TRANSPORT_IMPORTS = {
    "time",
    "tempfile",
    "mlflow.artifacts",
    "mlflow.entities",
    "mlflow.exceptions",
    "mlflow.tracking.artifact_utils",
    "mlflow.utils.uri",
}


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, check=True, capture_output=True
    )


def _require_training_commit(training_sha: str | None) -> None:
    if training_sha is None or re.fullmatch(r"[0-9a-fA-F]{40}", training_sha) is None:
        raise ValueError("An exact 40-character training Git revision is required.")
    _git("cat-file", "-e", f"{training_sha}^{{commit}}")


def _precomputed_path(config: DictConfig) -> bool:
    """Limit the compatibility exception to the original flat embedding graph."""
    targets = {
        "ml": {"ml.models.MammaprintModule", "ml.models.module.MammaprintModule"},
        "ml.encoder": {
            "ml.models.encoders.IdentityEncoder",
            "ml.models.encoders.identity.IdentityEncoder",
        },
        "ml.aggregator": {
            f"ml.models.aggregators.{name}"
            for name in ("MeanPool", "MaxPool", "AttentionMIL", "TransformerMIL")
        }
        | {
            f"ml.models.aggregators.{module}.{name}"
            for module, name in (
                ("mean", "MeanPool"),
                ("max", "MaxPool"),
                ("attention", "AttentionMIL"),
                ("transformer", "TransformerMIL"),
            )
        },
        "ml.head": {
            "ml.models.heads.LinearHead",
            "ml.models.heads.linear.LinearHead",
            "ml.models.heads.MLPHead",
            "ml.models.heads.mlp.MLPHead",
        },
        "datamodule": {"ml.data.DataModule", "ml.data.datamodule.DataModule"},
    }
    for key, allowed in targets.items():
        if OmegaConf.select(config, key + "._target_") not in allowed:
            return False
    activation = OmegaConf.select(config, "ml.output_activation")
    if activation is not None and (
        not isinstance(activation, DictConfig)
        or activation.get("_target_") != "torch.nn.Identity"
    ):
        return False
    for split in ("train", "val", "test"):
        dataset = OmegaConf.select(config, f"datamodule.{split}")
        if (
            not isinstance(dataset, DictConfig)
            or dataset.get("_target_")
            not in {
                "ml.data.SingleScaleDataset",
                "ml.data.datasets.single_scale.SingleScaleDataset",
            }
            or dataset.get("split") != split
            or not isinstance(dataset.get("levels"), DictConfig)
            or len(dataset.levels) != 1
        ):
            return False
    return True


def _scientific_source(source: str) -> str:
    """Compare every non-transport AST node, permitting only reviewed transfers.

    This includes label filtering, URI selection, slide intersection, their
    imports/constants and all module-level statements. Comparing function bodies
    alone would miss a changed dependency or a later name rebinding.
    """
    transport = []
    scientific = []
    for node in ast.parse(source).body:
        is_transport = (
            (isinstance(node, ast.FunctionDef) and node.name in _TRANSPORT_FUNCTIONS)
            or (isinstance(node, ast.ImportFrom) and node.module in _TRANSPORT_IMPORTS)
            or (
                isinstance(node, ast.Import)
                and all(alias.name in _TRANSPORT_IMPORTS for alias in node.names)
            )
        )
        (transport if is_transport else scientific).append(node)
    transfer_ast = ast.dump(
        ast.Module(body=transport, type_ignores=[]), include_attributes=False
    )
    if hashlib.sha256(transfer_ast.encode()).hexdigest() not in _REVIEWED_TRANSPORT:
        raise ValueError("Unreviewed artifact transfer implementation.")
    return ast.dump(
        ast.Module(body=scientific, type_ignores=[]), include_attributes=False
    )


def verify_training_revision(
    training_sha: str | None, *, config: DictConfig | None = None
) -> None:
    """Require exact sources or a reviewed precomputed-path compatibility check.

    Without config, retain the conservative whole-model/data comparison. With a
    resolved config, permit only reviewed artifact-transfer changes for the
    supported flat precomputed embedding graph. All other model/data/typing
    sources and scientific source helpers must match, including data modules
    eagerly imported by the package even when their datasets are not selected.
    Missing historical objects must be fetched explicitly; this never downloads
    code or guesses a revision from a mutable branch name.
    """
    try:
        _require_training_commit(training_sha)
        try:
            _git(
                "diff",
                "--exit-code",
                training_sha,
                "--",
                "ml/models",
                "ml/data",
                "ml/typing.py",
            )
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1 or config is None or not _precomputed_path(config):
                raise
        _git(
            "diff",
            "--exit-code",
            training_sha,
            "--",
            "ml/models",
            "ml/typing.py",
            "ml/data",
            ":(exclude)ml/data/datasets/_sources.py",
        )
        path = "ml/data/datasets/_sources.py"
        historical = _git("show", f"{training_sha}:{path}").stdout.decode()
        if _scientific_source(historical) != _scientific_source(
            (REPOSITORY / path).read_text()
        ):
            raise ValueError("Scientific data helpers differ from training.")
    except (subprocess.CalledProcessError, OSError, SyntaxError, ValueError) as exc:
        # Invalid/missing provenance remains a caller error as before.
        if (
            training_sha is None
            or re.fullmatch(r"[0-9a-fA-F]{40}", training_sha) is None
        ):
            raise ValueError(
                "An exact 40-character training Git revision is required."
            ) from exc
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
    verify_training_revision(revision, config=config)
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
