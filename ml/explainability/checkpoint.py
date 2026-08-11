"""Recover a trained MIL checkpoint and the Hydra configuration that produced it.

The explanation job is intentionally separate from training, but it must rebuild the
same encoder/aggregator/head graph before loading weights.  MLflow runs logged by
``rationai.mlkit.autolog`` contain the original Hydra command-line overrides in
``configs/hydra.yaml``; this module downloads and sanitises those overrides without
ever resuming or modifying the source run.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn


HYDRA_ARTIFACT = "configs/hydra.yaml"
SOURCE_CONFIG_ARTIFACT = "configs/config-resolved.yaml"
CHECKPOINT_FILENAME = "checkpoint.ckpt"
BEST_CHECKPOINT_DIRECTORY = "checkpoints/best"

# These values describe the original training invocation or identify its logger.
# Replaying them would either launch a training/test stage or append to the source
# run.  Model, data, task, seed, and trainer overrides are retained because they may
# affect the exact graph or dataset composition.
DEFAULT_DROP_OVERRIDE_KEYS = frozenset(
    {
        "checkpoint",
        "logger.run_id",
        "logger.run_name",
        "metadata.run_name",
        "mode",
        "test_after_fit",
    }
)
DEFAULT_DROP_OVERRIDE_PREFIXES = ("hydra.",)


class CheckpointError(RuntimeError):
    """Base class for explanation checkpoint/configuration failures."""


class CheckpointResolutionError(CheckpointError):
    """Raised when a local or MLflow checkpoint cannot be resolved exactly."""


class CheckpointLoadError(CheckpointError):
    """Raised when a checkpoint cannot be loaded strictly into the recovered graph."""


class SourceConfigMismatchError(CheckpointError):
    """Raised when replayed inference-critical config differs from the source run."""


class ArtifactClient(Protocol):
    """Small subset of :class:`mlflow.MlflowClient` used by this module."""

    def download_artifacts(
        self, run_id: str, path: str, dst_path: str | None = None
    ) -> str: ...


@dataclass(frozen=True)
class MlflowArtifactReference:
    """Components of an MLflow proxy artifact URI."""

    experiment_id: str
    run_id: str
    artifact_path: str


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A concrete local checkpoint plus optional source-run identity."""

    path: Path
    requested_uri: str
    source_run_id: str | None = None
    source_experiment_id: str | None = None
    source_artifact_path: str | None = None


@dataclass(frozen=True)
class CheckpointBootstrap:
    """All reproducibility inputs needed to instantiate an explanation job."""

    checkpoint: ResolvedCheckpoint
    logged_overrides: tuple[str, ...]
    caller_overrides: tuple[str, ...]
    effective_overrides: tuple[str, ...]
    hydra_config_path: Path | None
    source_config_path: Path | None
    checkpoint_sha256: str
    source_config_sha256: str | None
    override_hash: str
    source_hash: str

    @property
    def checkpoint_path(self) -> Path:
        """Shortcut used by the inference entrypoint."""
        return self.checkpoint.path

    @property
    def source_run_id(self) -> str | None:
        """The immutable training run, or ``None`` for a local checkpoint."""
        return self.checkpoint.source_run_id

    def provenance(self) -> dict[str, Any]:
        """Return JSON-ready checkpoint/config provenance for the manifest."""
        return {
            "checkpoint_uri": self.checkpoint.requested_uri,
            "checkpoint_path": str(self.checkpoint.path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_run_id": self.checkpoint.source_run_id,
            "source_experiment_id": self.checkpoint.source_experiment_id,
            "source_artifact_path": self.checkpoint.source_artifact_path,
            "source_config_artifact": (
                SOURCE_CONFIG_ARTIFACT if self.source_config_path is not None else None
            ),
            "source_config_sha256": self.source_config_sha256,
            "logged_overrides": list(self.logged_overrides),
            "caller_overrides": list(self.caller_overrides),
            "effective_overrides": list(self.effective_overrides),
            "override_hash": self.override_hash,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class CheckpointLoadInfo:
    """Useful non-weight metadata returned after a strict Lightning load."""

    state_dict_keys: int
    epoch: int | None
    global_step: int | None


def parse_mlflow_artifact_uri(uri: str) -> MlflowArtifactReference:
    """Parse ``mlflow-artifacts:/<experiment>/<run>/artifacts/<path>``.

    MLflow proxy artifact URIs have no authority/host component.  Parsing the URI
    structurally (rather than looking for a 32-character substring) makes source-run
    recovery independent of the experiment id and validates that the requested path
    really belongs to the run's artifact root.
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "mlflow-artifacts":
        raise ValueError(f"Not an mlflow-artifacts URI: {uri!r}")
    if parsed.netloc:
        raise ValueError(
            "mlflow-artifacts URIs must use the proxy path form without a host: "
            f"{uri!r}"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"Query strings and fragments are not valid artifact URIs: {uri!r}"
        )

    raw_parts = [part for part in parsed.path.split("/") if part]
    parts = [unquote(part) for part in raw_parts]
    if any(part in {".", ".."} or "/" in part or "\\" in part for part in parts):
        raise ValueError(f"Unsafe path component in MLflow artifact URI: {uri!r}")
    if len(parts) < 4 or parts[2] != "artifacts":
        raise ValueError(
            "Expected mlflow-artifacts:/<experiment-id>/<run-id>/artifacts/<path>; "
            f"got {uri!r}"
        )

    experiment_id, run_id = parts[:2]
    artifact_path = "/".join(parts[3:]).rstrip("/")
    if not experiment_id or not run_id or not artifact_path:
        raise ValueError(f"Incomplete MLflow artifact URI: {uri!r}")
    return MlflowArtifactReference(experiment_id, run_id, artifact_path)


def source_run_id_from_uri(uri: str) -> str:
    """Return the source run id from a validated MLflow artifact URI."""
    return parse_mlflow_artifact_uri(uri).run_id


def _new_mlflow_client(tracking_uri: str | None) -> ArtifactClient:
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _checkpoint_artifact_path(artifact_path: str) -> str:
    normalised = artifact_path.rstrip("/")
    if normalised.endswith(f"/{BEST_CHECKPOINT_DIRECTORY}") or normalised == (
        BEST_CHECKPOINT_DIRECTORY
    ):
        return f"{normalised}/{CHECKPOINT_FILENAME}"
    if PurePosixPath(normalised).name == CHECKPOINT_FILENAME:
        return normalised
    raise CheckpointResolutionError(
        "The MLflow URI must point to checkpoint.ckpt or a directory ending in "
        f"{BEST_CHECKPOINT_DIRECTORY!r}; got artifact path {artifact_path!r}."
    )


def _validate_local_checkpoint(path: Path, requested: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_dir():
        candidate = candidate / CHECKPOINT_FILENAME
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise CheckpointResolutionError(
            f"Checkpoint {requested!r} did not resolve to an existing file "
            f"({candidate})."
        )
    if candidate.suffix != ".ckpt":
        raise CheckpointResolutionError(
            f"Checkpoint must be a .ckpt file; resolved {candidate}."
        )
    return candidate


def resolve_checkpoint(
    checkpoint_uri: str | Path,
    destination: str | Path,
    *,
    tracking_uri: str | None = None,
    client: ArtifactClient | None = None,
) -> ResolvedCheckpoint:
    """Resolve a local path or supported MLflow URI to exactly one checkpoint file."""
    requested = str(checkpoint_uri)
    if requested.startswith("mlflow-artifacts:"):
        try:
            reference = parse_mlflow_artifact_uri(requested)
        except ValueError as error:
            raise CheckpointResolutionError(str(error)) from error
        artifact_path = _checkpoint_artifact_path(reference.artifact_path)
        destination_path = Path(destination).expanduser().resolve()
        destination_path.mkdir(parents=True, exist_ok=True)
        artifact_client = client or _new_mlflow_client(tracking_uri)
        try:
            downloaded = artifact_client.download_artifacts(
                reference.run_id, artifact_path, str(destination_path)
            )
        except Exception as error:
            raise CheckpointResolutionError(
                f"Could not download {artifact_path!r} from MLflow run "
                f"{reference.run_id}: {error}"
            ) from error
        local_path = _validate_local_checkpoint(Path(downloaded), requested)
        return ResolvedCheckpoint(
            path=local_path,
            requested_uri=requested,
            source_run_id=reference.run_id,
            source_experiment_id=reference.experiment_id,
            source_artifact_path=artifact_path,
        )

    parsed = urlsplit(requested)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise CheckpointResolutionError(
                f"Only local file:// checkpoint URIs are supported; got {requested!r}."
            )
        local = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise CheckpointResolutionError(
            "Checkpoint must be a local path, file:// URI, or mlflow-artifacts URI; "
            f"got {requested!r}."
        )
    else:
        local = Path(requested)
    return ResolvedCheckpoint(
        path=_validate_local_checkpoint(local, requested), requested_uri=requested
    )


def override_key(override: str) -> str:
    """Return a comparable Hydra key for add/change/delete override forms."""
    value = override.strip()
    if not value:
        raise ValueError("Hydra overrides cannot be empty.")
    value = value.lstrip("+~")
    key = value.split("=", 1)[0].strip()
    if not key:
        raise ValueError(f"Cannot identify a Hydra key in override {override!r}.")
    return key


def filter_training_overrides(
    overrides: Sequence[str],
    *,
    drop_keys: frozenset[str] = DEFAULT_DROP_OVERRIDE_KEYS,
    drop_prefixes: tuple[str, ...] = DEFAULT_DROP_OVERRIDE_PREFIXES,
) -> tuple[str, ...]:
    """Remove source-run and training-stage controls while retaining model/data state."""
    kept: list[str] = []
    for override in overrides:
        key = override_key(override)
        if key in drop_keys or key.startswith(drop_prefixes):
            continue
        kept.append(override)
    return tuple(kept)


def merge_overrides(
    recovered: Sequence[str], caller: Sequence[str] = ()
) -> tuple[str, ...]:
    """Apply caller overrides last, replacing matching recovered keys deterministically."""
    merged = list(recovered)
    for override in caller:
        key = override_key(override)
        merged = [existing for existing in merged if override_key(existing) != key]
        merged.append(override)
    return tuple(merged)


def recover_hydra_overrides(
    client: ArtifactClient, run_id: str, destination: str | Path
) -> tuple[tuple[str, ...], Path]:
    """Download the source run's Hydra artifact and return safe training overrides."""
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = client.download_artifacts(
            run_id, HYDRA_ARTIFACT, str(destination_path)
        )
    except Exception as error:
        raise CheckpointResolutionError(
            f"Could not download {HYDRA_ARTIFACT!r} from MLflow run {run_id}: {error}"
        ) from error

    hydra_path = Path(downloaded).resolve()
    if not hydra_path.is_file():
        raise CheckpointResolutionError(
            f"MLflow returned no file for run {run_id} artifact {HYDRA_ARTIFACT!r}."
        )
    try:
        hydra_config = OmegaConf.load(hydra_path)
    except Exception as error:
        raise CheckpointResolutionError(
            f"Could not read source Hydra config {hydra_path}: {error}"
        ) from error

    task = OmegaConf.select(hydra_config, "overrides.task")
    if task is None:
        task = OmegaConf.select(hydra_config, "hydra.overrides.task")
    if not isinstance(task, (list, tuple, ListConfig)) or not task:
        raise CheckpointResolutionError(
            f"Run {run_id} has no Hydra task overrides in {HYDRA_ARTIFACT!r}; "
            "the model/data graph cannot be reconstructed automatically."
        )
    if not all(isinstance(item, str) for item in task):
        raise CheckpointResolutionError(
            f"Run {run_id} has non-string Hydra task overrides in {HYDRA_ARTIFACT!r}."
        )
    return filter_training_overrides(tuple(task)), hydra_path


def recover_source_config(
    client: ArtifactClient, run_id: str, destination: str | Path
) -> Path:
    """Download the immutable resolved training config used by the source run."""
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = client.download_artifacts(
            run_id, SOURCE_CONFIG_ARTIFACT, str(destination_path)
        )
    except Exception as error:
        raise CheckpointResolutionError(
            f"Could not download {SOURCE_CONFIG_ARTIFACT!r} from MLflow run "
            f"{run_id}: {error}"
        ) from error
    source_path = Path(downloaded).resolve()
    if not source_path.is_file():
        raise CheckpointResolutionError(
            f"MLflow returned no file for run {run_id} artifact "
            f"{SOURCE_CONFIG_ARTIFACT!r}."
        )
    try:
        loaded = OmegaConf.load(source_path)
    except Exception as error:
        raise CheckpointResolutionError(
            f"Could not read source resolved config {source_path}: {error}"
        ) from error
    if not isinstance(loaded, DictConfig):
        raise CheckpointResolutionError(
            f"Source resolved config {source_path} is not a mapping."
        )
    return source_path


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading a potentially large checkpoint into memory twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Any) -> str:
    """Return a deterministic SHA-256 for an OmegaConf config or JSON-like value."""
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=False)

    def json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(f"Cannot hash config value of type {type(value).__name__}.")

    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bootstrap_checkpoint(
    checkpoint_uri: str | Path,
    destination: str | Path,
    *,
    caller_overrides: Sequence[str] = (),
    tracking_uri: str | None = None,
    client: ArtifactClient | None = None,
) -> CheckpointBootstrap:
    """Resolve weights and reconstruct the effective explanation configuration.

    A local checkpoint has no recoverable MLflow run, so its effective overrides are
    exactly ``caller_overrides``.  MLflow checkpoints always replay the logged model
    and data overrides first, followed by caller overrides.
    """
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_client = client
    if str(checkpoint_uri).startswith("mlflow-artifacts:") and artifact_client is None:
        artifact_client = _new_mlflow_client(tracking_uri)

    resolved = resolve_checkpoint(
        checkpoint_uri,
        root / "checkpoint",
        tracking_uri=tracking_uri,
        client=artifact_client,
    )
    logged: tuple[str, ...] = ()
    hydra_path: Path | None = None
    source_config_path: Path | None = None
    source_config_hash: str | None = None
    if resolved.source_run_id is not None:
        if artifact_client is None:  # defensive; remote resolution creates one above
            raise AssertionError(
                "MLflow checkpoint resolved without an artifact client."
            )
        logged, hydra_path = recover_hydra_overrides(
            artifact_client, resolved.source_run_id, root / "config"
        )
        source_config_path = recover_source_config(
            artifact_client, resolved.source_run_id, root / "source-config"
        )
        source_config_hash = file_sha256(source_config_path)

    caller = tuple(caller_overrides)
    effective = merge_overrides(logged, caller)
    checkpoint_hash = file_sha256(resolved.path)
    override_hash = stable_config_hash(list(effective))
    source_hash = stable_config_hash(
        {
            "checkpoint_sha256": checkpoint_hash,
            "override_hash": override_hash,
            "source_config_sha256": source_config_hash,
            "source_experiment_id": resolved.source_experiment_id,
            "source_run_id": resolved.source_run_id,
        }
    )
    return CheckpointBootstrap(
        checkpoint=resolved,
        logged_overrides=logged,
        caller_overrides=caller,
        effective_overrides=effective,
        hydra_config_path=hydra_path,
        source_config_path=source_config_path,
        checkpoint_sha256=checkpoint_hash,
        source_config_sha256=source_config_hash,
        override_hash=override_hash,
        source_hash=source_hash,
    )


def compose_training_config(
    overrides: Sequence[str],
    *,
    config_dir: str | Path | None = None,
    config_name: str = "ml",
) -> DictConfig:
    """Compose the original training graph outside a Hydra-decorated entrypoint."""
    from hydra import compose, initialize_config_dir

    # ``ml.train`` normally registers this resolver at import time.  The standalone
    # explainer composes without importing the training entrypoint, so mirror its
    # semantics here.  A recovered explicit seed still takes precedence.
    if not OmegaConf.has_resolver("random_seed"):
        OmegaConf.register_new_resolver(
            "random_seed", lambda: random.randint(0, 2**31), use_cache=True
        )

    resolved_config_dir = (
        Path(config_dir).expanduser().resolve()
        if config_dir is not None
        else Path(__file__).resolve().parents[2] / "configs"
    )
    with initialize_config_dir(
        config_dir=str(resolved_config_dir), version_base=None, job_name="explain"
    ):
        return compose(config_name=config_name, overrides=list(overrides))


INFERENCE_CRITICAL_CONFIG_PATHS = (
    "seed",
    "label_mode",
    "feature_dim",
    "dataset",
    "datamodule",
    "ml._target_",
    "ml.encoder",
    "ml.aggregator",
    "ml.head",
    "ml.output_activation",
)


def validate_replayed_source_config(
    replayed: DictConfig,
    source_config_path: str | Path,
) -> str:
    """Reject source-run drift in the model, data, task, or geometry config.

    Hydra overrides are replayed so caller overrides continue to support config
    groups.  This comparison prevents the same override strings from silently
    resolving to changed data cards or architecture defaults on a newer branch.
    The returned hash covers the compared source configuration subset.
    """
    source = OmegaConf.load(Path(source_config_path))
    if not isinstance(source, DictConfig):
        raise SourceConfigMismatchError("The source resolved config is not a mapping.")

    source_values: dict[str, Any] = {}
    replayed_values: dict[str, Any] = {}
    missing: list[str] = []
    for path in INFERENCE_CRITICAL_CONFIG_PATHS:
        source_value = OmegaConf.select(source, path, default=None)
        replayed_value = OmegaConf.select(replayed, path, default=None)
        if source_value is None or replayed_value is None:
            missing.append(path)
            continue
        source_values[path] = _plain_config_value(source_value)
        replayed_values[path] = _plain_config_value(replayed_value)
    if missing:
        raise SourceConfigMismatchError(
            "Cannot verify the source run because inference-critical config keys are "
            f"missing: {', '.join(missing)}."
        )

    mismatches = [
        path
        for path in INFERENCE_CRITICAL_CONFIG_PATHS
        if stable_config_hash(source_values[path])
        != stable_config_hash(replayed_values[path])
    ]
    if mismatches:
        raise SourceConfigMismatchError(
            "Replaying the source run's Hydra overrides on this checkout changed "
            "inference-critical configuration. Refusing to explain a mismatched "
            f"model/data graph; differing keys: {', '.join(mismatches)}."
        )
    return stable_config_hash(source_values)


def _plain_config_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def load_lightning_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadInfo:
    """Strictly load a Lightning ``state_dict`` and place the module in eval mode."""
    path = _validate_local_checkpoint(Path(checkpoint_path), str(checkpoint_path))
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointLoadError(
            f"Could not read Lightning checkpoint {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointLoadError(
            f"Lightning checkpoint {path} is {type(payload).__name__}, not a mapping."
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointLoadError(
            f"Lightning checkpoint {path} has no non-empty 'state_dict' mapping."
        )
    if not all(isinstance(key, str) for key in state_dict):
        raise CheckpointLoadError(
            f"Lightning checkpoint {path} contains non-string state_dict keys."
        )
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointLoadError(
            "Checkpoint weights do not exactly match the recovered model graph: "
            f"{error}"
        ) from error
    module.eval()
    return CheckpointLoadInfo(
        state_dict_keys=len(state_dict),
        epoch=_optional_int(payload.get("epoch")),
        global_step=_optional_int(payload.get("global_step")),
    )


__all__ = [
    "BEST_CHECKPOINT_DIRECTORY",
    "CHECKPOINT_FILENAME",
    "HYDRA_ARTIFACT",
    "INFERENCE_CRITICAL_CONFIG_PATHS",
    "SOURCE_CONFIG_ARTIFACT",
    "CheckpointBootstrap",
    "CheckpointError",
    "CheckpointLoadError",
    "CheckpointLoadInfo",
    "CheckpointResolutionError",
    "MlflowArtifactReference",
    "ResolvedCheckpoint",
    "SourceConfigMismatchError",
    "bootstrap_checkpoint",
    "compose_training_config",
    "file_sha256",
    "filter_training_overrides",
    "load_lightning_checkpoint",
    "merge_overrides",
    "override_key",
    "parse_mlflow_artifact_uri",
    "recover_hydra_overrides",
    "recover_source_config",
    "resolve_checkpoint",
    "source_run_id_from_uri",
    "stable_config_hash",
    "validate_replayed_source_config",
]
