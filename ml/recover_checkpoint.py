"""Retrain selected checkpoints under their original training protocol.

The default reads metadata only. Heavy training dependencies are imported inside
execution so ``python -S -m ml.recover_checkpoint --help`` stays offline.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


RESOLVED_CONFIG = "configs/config-resolved.yaml"
CHECKPOINT = "checkpoints/best/checkpoint.ckpt"
MLMODEL = "checkpoints/best/MLmodel"
REPOSITORY = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-f]{40}")
_RUN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_CHECKPOINT_TARGET = "lightning.pytorch.callbacks.ModelCheckpoint"
# Only the native precomputed graph is supported; the scientific source gate is
# additionally required before instantiation. No generic target prefixes.
_ALLOWED_TARGETS = {
    "ml.models.MammaprintModule",
    "ml.models.encoders.IdentityEncoder",
    "ml.models.aggregators.MeanPool",
    "ml.models.aggregators.MaxPool",
    "ml.models.aggregators.AttentionMIL",
    "ml.models.aggregators.TransformerMIL",
    "ml.models.aggregators.SpatialTransformerMIL",
    "ml.models.heads.LinearHead",
    "ml.models.heads.MLPHead",
    "ml.models.losses.JointLoss",
    "torch.nn.Identity",
    "torch.nn.Softmax",
    "torch.nn.BCEWithLogitsLoss",
    "torch.nn.MSELoss",
    "torch.nn.CrossEntropyLoss",
    "torch.optim.Adam",
    "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "ml.metrics.BinaryLogitMetric",
    "ml.metrics.ColumnMetric",
    "ml.metrics.thresholded_f1",
    "ml.metrics.thresholded_specificity",
    "ml.metrics.thresholded_precision",
    "ml.metrics.thresholded_recall",
    "torchmetrics.MeanSquaredError",
    "torchmetrics.MeanAbsoluteError",
    "ml.data.DataModule",
    "ml.data.SingleScaleDataset",
    "ml.data.SpatialScaleDataset",
    "rationai.mlkit.lightning.Trainer",
    "rationai.mlkit.lightning.loggers.MLFlowLogger",
    "ml.loggers.MLFlowLogger",
    _CHECKPOINT_TARGET,
    "lightning.pytorch.callbacks.EarlyStopping",
    "lightning.pytorch.callbacks.LearningRateMonitor",
} | {
    f"torchmetrics.classification.Binary{n}"
    for n in ("Accuracy", "AUROC", "F1Score", "Precision", "Recall", "Specificity")
}


def _safe_text(value: str) -> str:
    if (
        not isinstance(value, str)
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or "${" in value
    ):
        raise ValueError(
            "Control characters and configuration interpolation are forbidden."
        )
    return value


def _sha(value: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("An exact lowercase 40-character Git SHA is required.")
    return value


def normalize_tracking_uri(value: str) -> str:
    parsed = urlsplit(_safe_text(value))
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Use an explicit credentials-free HTTP(S) tracking URI without query/fragment."
        )
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid tracking endpoint port.") from exc
    if (
        "%" in parsed.netloc
        or "\\" in value
        or any(p in {".", ".."} for p in parsed.path.split("/"))
    ):
        raise ValueError("Invalid tracking endpoint.")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


@dataclass(frozen=True)
class SourceIdentity:
    tracking_uri: str
    run_id: str
    experiment_id: str | None = None


def parse_source(value: str, tracking_uri: str) -> SourceIdentity:
    uri = normalize_tracking_uri(tracking_uri)
    _safe_text(value)
    if _RUN.fullmatch(value):
        return SourceIdentity(uri, value)
    parsed = urlsplit(value)
    if parsed.query or parsed.username or parsed.password:
        raise ValueError("Source URL must not contain credentials or a query.")
    fragment = parsed.fragment
    path = fragment if fragment else parsed.path
    match = re.fullmatch(
        r"/?experiments/(\d+)/runs/([A-Za-z0-9_-]{1,128})"
        r"(?:/(?:model-metrics|artifacts|overview|system-metrics))?/?",
        path,
    )
    if not match:
        raise ValueError("Expected a safe MLflow run ID or an experiment/run URL.")
    base_path = parsed.path if fragment else ""
    base = normalize_tracking_uri(
        urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
    )
    if base != uri:
        raise ValueError("Source URL and explicit tracking endpoint disagree.")
    return SourceIdentity(uri, match[2], match[1])


@dataclass(frozen=True)
class MetadataLimits:
    config_bytes: int = 1024 * 1024
    page_size: int = 1000
    max_records: int = 200_000
    history_bytes: int = 32 * 1024 * 1024
    max_pages: int = 400

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 1 for v in vars(self).values())
            or self.page_size > 1000
        ):
            raise ValueError(
                "Metadata limits must be positive integers; page size is at most 1000."
            )


class PagedMetricReader:
    """Isolate MLflow's paged REST-store API; never call its unbounded SDK helper.

    MLflow may return an oversized page despite max_results: reject it, never
    select from a truncated response. Limits are shared across monitor + epoch.
    """

    def __init__(self, store: Any, limits: MetadataLimits):
        self.store, self.limits = store, limits
        self.records = self.bytes = self.pages = 0

    @classmethod
    def from_client(cls, client: Any, limits: MetadataLimits):
        from mlflow.store.tracking.rest_store import RestStore

        store = client._tracking_client.store
        if not isinstance(store, RestStore):
            raise ValueError(
                "Bounded metadata planning requires MLflow's paged REST tracking store."
            )
        return cls(store, limits)

    def read(self, run_id: str, key: str) -> list[Any]:
        token, seen, result = None, set(), []
        while True:
            self.pages += 1
            if self.pages > self.limits.max_pages:
                raise ValueError("Metric page limit exceeded; history is incomplete.")
            page = self.store.get_metric_history(
                run_id, key, max_results=self.limits.page_size, page_token=token
            )
            if len(page) > self.limits.page_size:
                raise ValueError(
                    "Server exceeded metric page limit; refusing incomplete history."
                )
            for item in page:
                self.records += 1
                self.bytes += len(
                    json.dumps(
                        [item.step, item.timestamp, item.value], allow_nan=True
                    ).encode()
                )
                if (
                    self.records > self.limits.max_records
                    or self.bytes > self.limits.history_bytes
                ):
                    raise ValueError(
                        "Metric history limit exceeded; refusing truncated history."
                    )
                result.append(item)
            token = getattr(page, "token", None)
            if not token:
                return result
            if not isinstance(token, str) or token in seen or len(token) > 4096:
                raise ValueError("Invalid/repeated metric pagination token.")
            seen.add(token)

    def report(self) -> dict:
        return {
            "records": self.records,
            "bytes": self.bytes,
            "pages": self.pages,
            "limits": vars(self.limits),
            "complete": True,
        }


@dataclass
class SourceEvidence:
    identity: SourceIdentity
    experiment_id: str
    source_training_sha: str
    config_bytes: bytes
    config: dict
    monitor_history: list
    epoch_history: list
    history_read: dict


def _selection(
    config: dict, monitor: str | None, mode: str | None
) -> tuple[str, str, str, dict]:
    callbacks = config.get("trainer", {}).get("callbacks", {})
    if not isinstance(callbacks, dict):
        raise ValueError("Only named native callback dictionaries are supported.")
    checkpoints = [
        (k, v)
        for k, v in callbacks.items()
        if isinstance(v, dict) and v.get("_target_") == _CHECKPOINT_TARGET
    ]
    if len(checkpoints) != 1:
        raise ValueError("Exactly one original ModelCheckpoint is required.")
    name, checkpoint = checkpoints[0]
    original_monitor, original_mode = checkpoint.get("monitor"), checkpoint.get("mode")
    if (
        not isinstance(original_monitor, str)
        or not original_monitor.startswith("valid/")
        or original_mode not in {"min", "max"}
    ):
        raise ValueError(
            "Original checkpoint must declare a validation monitor and mode."
        )
    if (monitor is None) != (mode is None):
        raise ValueError("--monitor and --mode must be supplied together.")
    selected, direction = (
        (monitor, mode) if monitor is not None else (original_monitor, original_mode)
    )
    if (
        not selected
        or not re.fullmatch(r"valid/[A-Za-z0-9_./-]+", selected)
        or direction not in {"min", "max"}
    ):
        raise ValueError(
            "Selection requires a validation-only metric and min/max mode."
        )
    return selected, direction, name, checkpoint


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("Resolved configuration must be a mapping.")

    def walk(node):
        if isinstance(node, dict):
            if "_target_" in node and node["_target_"] not in _ALLOWED_TARGETS:
                raise ValueError("Unsupported native configuration target.")
            if any(k in node for k in ("_args_", "_convert_")):
                raise ValueError("Unsupported Hydra instantiation directive.")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            if "${" in node or any(
                (ord(c) < 32 and c not in "\n\r\t") or ord(c) == 127 for c in node
            ):
                raise ValueError(
                    "Unresolved interpolation/control characters in source config."
                )
        elif isinstance(node, float) and not math.isfinite(node):
            raise ValueError("Configuration numbers must be finite.")

    walk(config)
    if config.get("mode") != "fit" or config.get("checkpoint") is not None:
        raise ValueError(
            "Replay requires original fit from seeded initialization, without an initial checkpoint."
        )
    if type(config.get("seed")) is not int:
        raise ValueError("Resolved source must contain the original integer seed.")
    required = {
        "ml": "ml.models.MammaprintModule",
        "datamodule": "ml.data.DataModule",
        "trainer": "rationai.mlkit.lightning.Trainer",
    }
    if any(
        config.get(k, {}).get("_target_") != target for k, target in required.items()
    ):
        raise ValueError("Unsupported native model/datamodule/trainer target.")


def parse_config_bytes(raw: bytes) -> dict:
    import yaml

    # Hydra's resolved export never needs YAML aliases. Refuse alias cycles or
    # expansion before walking/hash-canonicalizing the untrusted metadata.
    if any(
        isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
        for token in yaml.scan(raw)
    ):
        raise ValueError("Resolved config YAML aliases/anchors are unsupported.")
    config = yaml.safe_load(raw)
    _validate_config(config)
    return config


def read_source(
    client: Any,
    identity: SourceIdentity,
    *,
    reader: PagedMetricReader | None = None,
    limits: MetadataLimits | None = None,
    training_sha: str | None = None,
    monitor: str | None = None,
    mode: str | None = None,
) -> SourceEvidence:
    limits = limits or MetadataLimits()
    if training_sha is not None:
        _sha(training_sha)
    run = client.get_run(identity.run_id)
    experiment = str(run.info.experiment_id)
    if (
        run.info.run_id != identity.run_id
        or not re.fullmatch(r"\d+", experiment)
        or (identity.experiment_id is not None and identity.experiment_id != experiment)
    ):
        raise ValueError("Source run/experiment identity mismatch.")
    if run.info.status != "FINISHED" or run.info.lifecycle_stage != "active":
        raise ValueError("Source run must be FINISHED and active.")
    recorded = run.data.tags.get("mlflow.source.git.commit")
    if recorded and training_sha and recorded != training_sha:
        raise ValueError("Explicit training SHA conflicts with source tag.")
    revision = _sha(recorded or training_sha)
    entries = client.list_artifacts(identity.run_id, "configs")
    if len(entries) > 100:
        raise ValueError("Source config artifact listing exceeds metadata limit.")
    matches = [
        entry for entry in entries if entry.path == RESOLVED_CONFIG and not entry.is_dir
    ]
    if (
        len(matches) != 1
        or type(matches[0].file_size) is not int
        or not 0 < matches[0].file_size <= limits.config_bytes
    ):
        raise ValueError(
            "Resolved config artifact missing or exceeds metadata size limit."
        )
    with tempfile.TemporaryDirectory(prefix="checkpoint-metadata-") as directory:
        downloaded = Path(
            client.download_artifacts(identity.run_id, RESOLVED_CONFIG, directory)
        )
        if (
            downloaded.stat().st_size != matches[0].file_size
            or downloaded.stat().st_size > limits.config_bytes
        ):
            raise ValueError("Config readback size differs or exceeds metadata limit.")
        raw = downloaded.read_bytes()
    config = parse_config_bytes(raw)
    _validate_config(config)
    metric, _, _, _ = _selection(config, monitor, mode)
    reader = reader or PagedMetricReader.from_client(client, limits)
    history = reader.read(identity.run_id, metric)
    epochs = reader.read(identity.run_id, "epoch")
    return SourceEvidence(
        identity, experiment, revision, raw, config, history, epochs, reader.report()
    )


def _events(history: list, *, epoch: bool = False) -> list[tuple[int, int, float]]:
    events = set()
    for item in history:
        if (
            type(item.step) is not int
            or item.step < 0
            or type(item.timestamp) is not int
            or item.timestamp < 0
            or isinstance(item.value, bool)
            or not isinstance(item.value, (int, float))
            or not math.isfinite(item.value)
        ):
            raise ValueError(
                "History requires finite values and nonnegative integer steps/timestamps."
            )
        if epoch and (int(item.value) != item.value or item.value < 0):
            raise ValueError("History epoch must be a nonnegative integer.")
        events.add((item.step, item.timestamp, float(item.value)))
    if not events:
        raise ValueError("Complete monitor and epoch history are required.")
    return sorted(events, key=lambda event: (event[0], event[1]))


def plan_replay(
    evidence: SourceEvidence,
    *,
    implementation_sha: str,
    monitor: str | None = None,
    mode: str | None = None,
) -> dict:
    _sha(implementation_sha)
    _sha(evidence.source_training_sha)
    config = evidence.config
    _validate_config(config)
    metric, direction, checkpoint_name, checkpoint = _selection(config, monitor, mode)
    trainer = config["trainer"]
    if (
        trainer.get("check_val_every_n_epoch", 1) != 1
        or type(trainer.get("val_check_interval", 1.0)) is not float
        or trainer.get("val_check_interval", 1.0) != 1.0
        or trainer.get("fast_dev_run", False)
        or trainer.get("limit_val_batches", 1.0) == 0
        or trainer.get("enable_checkpointing", True) is False
        or trainer.get("barebones", False)
        or trainer.get("reload_dataloaders_every_n_epochs", 0) != 0
        or checkpoint.get("every_n_train_steps", 0) not in (None, 0)
        or checkpoint.get("train_time_interval") is not None
    ):
        raise ValueError(
            "Unsupported validation/checkpoint cadence; require one validation per epoch."
        )
    every = checkpoint.get("every_n_epochs", 1)
    every = 1 if every is None else every
    if type(every) is not int or every < 1 or checkpoint.get("save_top_k", 1) != 1:
        raise ValueError(
            "Require native best-only checkpoint cadence with positive every_n_epochs."
        )
    monitor_events = _events(evidence.monitor_history)
    epoch_events = _events(evidence.epoch_history, epoch=True)
    exact, by_step = {}, {}
    for step, timestamp, epoch in epoch_events:
        exact.setdefault((step, timestamp), set()).add(int(epoch))
        by_step.setdefault(step, set()).add(int(epoch))
    # Any chronological regression means a resumed/restarted session, not a
    # single seeded fit. Epoch's per-batch records may repeat harmlessly.
    previous_step = previous_epoch = -1
    for step, _, epoch in sorted(epoch_events, key=lambda event: (event[1], event[0])):
        if step < previous_step or epoch < previous_epoch:
            raise ValueError(
                "Epoch history is ambiguous or contains a session/step reset."
            )
        previous_step, previous_epoch = step, epoch
    aligned, seen_steps, fallback = [], set(), False
    for step, timestamp, value in monitor_events:
        if step == 0:
            raise ValueError(
                "Cannot distinguish a step-zero sanity metric from completed training."
            )
        if step in seen_steps:
            raise ValueError(
                "Multiple/conflicting validation events at one step are unsupported."
            )
        seen_steps.add(step)
        epochs = exact.get((step, timestamp))
        if epochs is None:
            fallback = True
            epochs = by_step.get(step, set())
        if len(epochs) != 1:
            raise ValueError(
                "Missing or ambiguous epoch alignment for validation event."
            )
        aligned.append((epochs.copy().pop(), step, timestamp, value))
    if [entry[0] for entry in aligned] != list(range(len(aligned))):
        raise ValueError(
            "Unsupported cadence, missing epochs or repeated validation/session reset."
        )
    candidates = [entry for entry in aligned if (entry[0] + 1) % every == 0]
    if not candidates:
        raise ValueError("No checkpoint-eligible validation event.")
    best = (min if direction == "min" else max)(entry[3] for entry in candidates)
    epoch, step, timestamp, score = next(
        entry for entry in candidates if entry[3] == best
    )
    max_epochs, max_steps = (
        trainer.get("max_epochs", 1000),
        trainer.get("max_steps", -1),
    )
    if max_epochs is None:
        max_epochs = 1000 if max_steps == -1 else -1
    if (max_epochs != -1 and (type(max_epochs) is not int or epoch >= max_epochs)) or (
        max_steps != -1 and (type(max_steps) is not int or step + 1 > max_steps)
    ):
        raise ValueError("Historical target exceeds original training horizon.")
    if max_epochs == -1 and max_steps == -1:
        raise ValueError("Original training must have a finite epoch or step limit.")
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "format_version": 2,
        "kind": "retrained",
        "training_policy": "original_horizon_and_early_stopping",
        "tracking_uri": evidence.identity.tracking_uri,
        "experiment_id": evidence.experiment_id,
        "source_run_id": evidence.identity.run_id,
        "source_training_sha": evidence.source_training_sha,
        "implementation_sha": implementation_sha,
        "source_config_sha256": hashlib.sha256(evidence.config_bytes).hexdigest(),
        "source_config_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "monitor": metric,
        "mode": direction,
        "historical_best_score": score,
        "historical_best_epoch": epoch,
        "historical_best_step": step,
        "historical_best_timestamp": timestamp,
        "trainer_max_epochs": max_epochs,
        "trainer_max_steps": max_steps,
        "checkpoint_callback": checkpoint_name,
        "selection_changed": monitor is not None,
        "epoch_alignment": "unambiguous_step" if fallback else "exact_step_timestamp",
        "history_read": evidence.history_read,
        "historical_test_metrics_revalidated": False,
        "reproducibility": "Seeded retraining; historical runtime and data bytes are unverified; exact lost weights are not guaranteed.",
    }


def _config_diff(original: Any, effective: Any, path: str = "") -> list[dict]:
    if isinstance(original, dict) and isinstance(effective, dict):
        changes = []
        for key in sorted(original.keys() | effective.keys()):
            name = f"{path}.{key}" if path else key
            if key not in original or key not in effective:
                changes.append(
                    {
                        "path": name,
                        "before": original.get(key),
                        "after": effective.get(key),
                    }
                )
            else:
                changes.extend(_config_diff(original[key], effective[key], name))
        return changes
    if type(original) is not type(effective) or original != effective:
        return [{"path": path, "before": original, "after": effective}]
    return []


def validate_effective_config(
    original: dict, effective: dict, plan: dict
) -> list[dict]:
    changes = _config_diff(original, effective)
    checkpoint = f"trainer.callbacks.{plan['checkpoint_callback']}"
    allowed = {
        "test_after_fit",
        "trainer.default_root_dir",
        f"{checkpoint}.dirpath",
        f"{checkpoint}.filename",
    }
    if plan["selection_changed"]:
        allowed |= {f"{checkpoint}.monitor", f"{checkpoint}.mode"}
    for change in changes:
        if not (change["path"].startswith("logger.") or change["path"] in allowed):
            raise ValueError(
                f"Effective configuration change outside allowlist: {change['path']}"
            )
    if (
        effective["ml"] != original["ml"]
        or effective["datamodule"] != original["datamodule"]
    ):
        raise ValueError("Scientific configuration differs outside the allowlist.")
    return changes


def effective_config(
    original: dict, plan: dict, output: Path, run_id: str
) -> tuple[dict, list[dict]]:
    config = copy.deepcopy(original)
    config["test_after_fit"] = True
    tags = dict(config.get("logger", {}).get("tags", {}))
    tags.update(
        {
            "recovery.source_run_id": plan["source_run_id"],
            "recovery.kind": "retrained",
            "recovery.source_training_sha": plan["source_training_sha"],
            "recovery.source_config_sha256": plan["source_config_sha256"],
            "recovery.implementation_sha": plan["implementation_sha"],
            "recovery.historical_test_metrics_revalidated": "false",
        }
    )
    config["logger"] = dict(
        config.get("logger", {}),
        _target_="ml.loggers.MLFlowLogger",
        tracking_uri=plan["tracking_uri"],
        run_id=run_id,
        run_name=f"checkpoint-recovery-{plan['source_run_id'][:8]}",
        save_dir=str(output),
        log_model="all",
        tags=tags,
    )
    config["trainer"]["default_root_dir"] = str(output)
    checkpoint = config["trainer"]["callbacks"][plan["checkpoint_callback"]]
    checkpoint.update(dirpath=str(output / "checkpoints"), filename="best")
    if plan["selection_changed"]:
        checkpoint.update(monitor=plan["monitor"], mode=plan["mode"])
    return config, validate_effective_config(original, config, plan)


def _git(*args: str, repository: Path = REPOSITORY) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout


def ensure_training_revision(sha: str) -> None:
    """Fetch only a missing recorded object, from the expected repository.

    Called only by explicit execution, never metadata planning. Historical code
    is not checked out or executed; the compatibility gate still follows.
    """
    _sha(sha)
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}")
        return
    except subprocess.CalledProcessError:
        pass
    origin = _git("remote", "get-url", "origin").strip().removesuffix(".git")
    if origin not in {
        "https://github.com/RationAI/mammaprint",
        "git@github.com:RationAI/mammaprint",
    }:
        raise ValueError("Refusing historical-object fetch from an unexpected origin.")
    _git("fetch", "--no-tags", "--no-write-fetch-head", "origin", sha)
    _git("cat-file", "-e", f"{sha}^{{commit}}")


def _training_ast(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            in {
                "rationai.mlkit.lightning.loggers",
                "ml.loggers",
            }
            and [(n.name, n.asname) for n in node.names] == [("MLFlowLogger", None)]
        ):
            node.module = "reviewed_logger_annotation_import"
    return ast.dump(tree, include_attributes=False)


def _dependency_closure(project: dict, lock: dict) -> dict[str, dict]:
    """Resolve the original base dependency graph on the current Python/platform.

    Extras added later are outside this graph. Ambiguous active lock candidates
    fail closed rather than selecting a convenient pin.
    """
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    packages = {}
    for package in lock["package"]:
        packages.setdefault(canonicalize_name(package["name"]), []).append(package)
    pending = []
    for declaration in project["project"]["dependencies"]:
        requirement = Requirement(declaration)
        if requirement.marker is None or requirement.marker.evaluate():
            if requirement.extras:
                raise ValueError(
                    "Original dependency extras require an explicitly supported closure."
                )
            pending.append({"name": canonicalize_name(requirement.name)})
    selected = {}
    while pending:
        dependency = pending.pop()
        if dependency.get("marker") and not Marker(dependency["marker"]).evaluate():
            continue
        name = canonicalize_name(dependency["name"])
        candidates = [
            p
            for p in packages.get(name, [])
            if (
                not dependency.get("version")
                or p.get("version") == dependency["version"]
            )
            and (
                not dependency.get("source") or p.get("source") == dependency["source"]
            )
            and (
                not p.get("resolution-markers")
                or any(Marker(m).evaluate() for m in p["resolution-markers"])
            )
        ]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous or missing original dependency pin: {name}")
        package = candidates[0]
        fingerprint = {
            "version": package.get("version"),
            "source": package.get("source", {}),
        }
        if name in selected:
            if selected[name] != fingerprint:
                raise ValueError(f"Conflicting original dependency pins: {name}")
            continue
        selected[name] = fingerprint
        pending.extend(package.get("dependencies", []))
    return selected


def verify_replay_provenance(
    plan: dict, config: dict, *, repository: Path = REPOSITORY
) -> dict:
    """Verify committed implementation, scientific sources and original pins.

    No checkout/fetch/install is performed. Missing historical objects require
    an explicit, identity-checked fetch outside this function.
    """
    import tomllib

    from omegaconf import OmegaConf
    from packaging.specifiers import SpecifierSet

    from ml.concept_explainability.model import verify_training_revision

    sha, implementation = (
        _sha(plan["source_training_sha"]),
        _sha(plan["implementation_sha"]),
    )
    if _git("rev-parse", "HEAD", repository=repository).strip() != implementation:
        raise ValueError("Current HEAD does not match the pinned implementation SHA.")
    if _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "ml",
        "uv.lock",
        "pyproject.toml",
        repository=repository,
    ):
        raise ValueError(
            "Training/recovery implementation and dependency files must be committed and clean."
        )
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}", repository=repository)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Historical training Git object is missing. Fetch that exact SHA after verifying origin identity."
        ) from exc
    verify_training_revision(sha, config=OmegaConf.create(config))
    if _git("diff", sha, "--", "ml/metrics.py", "ml/metrics", repository=repository):
        raise ValueError("Training metrics differ from historical source.")
    historical_train = _git("show", f"{sha}:ml/train.py", repository=repository)
    if _training_ast(historical_train) != _training_ast(
        (repository / "ml/train.py").read_text()
    ):
        raise ValueError(
            "Training loop differs beyond the reviewed logger annotation import."
        )
    historical_project_bytes = _git(
        "show", f"{sha}:pyproject.toml", repository=repository
    ).encode()
    historical_lock_bytes = _git(
        "show", f"{sha}:uv.lock", repository=repository
    ).encode()
    historical_project = tomllib.loads(historical_project_bytes.decode())
    historical_lock = tomllib.loads(historical_lock_bytes.decode())
    current_lock = tomllib.loads((repository / "uv.lock").read_text())
    python_version = ".".join(map(str, sys.version_info[:3]))
    if not SpecifierSet(historical_project["project"]["requires-python"]).contains(
        python_version
    ):
        raise ValueError("Runtime Python does not satisfy original project constraint.")
    if historical_lock.get("requires-python") != current_lock.get("requires-python"):
        raise ValueError("Original lock Python compatibility differs.")
    original = _dependency_closure(historical_project, historical_lock)
    current = _dependency_closure(historical_project, current_lock)
    if original != current:
        raise ValueError(
            "Historical training dependency closure pins/source identities changed."
        )
    installed = {}
    for name, pin in original.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != pin["version"]:
            raise ValueError(
                f"Installed dependency does not match original pin: {name}"
            )
        source = pin["source"]
        if "git" in source:
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            git_url = urlsplit(source["git"])
            expected_url = urlunsplit(
                (git_url.scheme, git_url.netloc, git_url.path, "", "")
            )
            if (
                direct.get("url", "").rstrip("/") != expected_url.rstrip("/")
                or direct.get("vcs_info", {}).get("commit_id") != git_url.fragment
            ):
                raise ValueError(f"Installed Git dependency source differs: {name}")
        elif source.get("registry") not in {"https://pypi.org/simple", None}:
            raise ValueError(f"Unsupported historical registry provenance: {name}")
        installed[name] = distribution.version
    import torch

    return {
        "source_training_sha": sha,
        "implementation_sha": implementation,
        "historical_project_sha256": hashlib.sha256(
            historical_project_bytes
        ).hexdigest(),
        "historical_lock_sha256": hashlib.sha256(historical_lock_bytes).hexdigest(),
        "current_lock_sha256": hashlib.sha256(
            (repository / "uv.lock").read_bytes()
        ).hexdigest(),
        "training_dependency_closure": original,
        "installed_versions": installed,
        "python": python_version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "historical_runtime_verified": False,
        "historical_data_hashes_verified": False,
    }


def result_manifest(
    plan: dict,
    *,
    recovery_run_id: str,
    score: float,
    checkpoint_epoch: int,
    checkpoint_global_step: int,
    checkpoint_sha256: str,
    checkpoint_size: int,
    test_metrics: dict,
) -> dict:
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("Recovered checkpoint score must be finite.")
    if (
        not _RUN.fullmatch(recovery_run_id)
        or recovery_run_id == plan["source_run_id"]
        or type(checkpoint_epoch) is not int
        or checkpoint_epoch < 0
        or (
            plan["trainer_max_epochs"] != -1
            and checkpoint_epoch >= plan["trainer_max_epochs"]
        )
        or type(checkpoint_global_step) is not int
        or checkpoint_global_step < 1
        or (
            plan["trainer_max_steps"] != -1
            and checkpoint_global_step > plan["trainer_max_steps"]
        )
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
        or type(checkpoint_size) is not int
        or checkpoint_size < 1
    ):
        raise ValueError(
            "Recovered checkpoint identity/metadata is invalid or exceeds the original training limits."
        )
    if not isinstance(test_metrics, dict) or not test_metrics:
        raise ValueError("A completed best-checkpoint test evaluation is required.")
    if any(
        not isinstance(key, str)
        or not key.startswith("test/")
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for key, value in test_metrics.items()
    ):
        raise ValueError("Test metrics must be finite test/... scalar values.")
    reached = (
        score <= plan["historical_best_score"]
        if plan["mode"] == "min"
        else score >= plan["historical_best_score"]
    )
    return dict(
        plan,
        status="completed",
        recovery_run_id=recovery_run_id,
        observed_best_score=score,
        historical_target_reached=reached,
        checkpoint_artifact_path=CHECKPOINT,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_global_step=checkpoint_global_step,
        test_evaluation={
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_global_step": checkpoint_global_step,
            "metrics": test_metrics,
        },
        model_artifact_path=MLMODEL,
        historical_test_metrics_revalidated=False,
    )


def _reject_inherited_context() -> None:
    import os

    if os.environ.get("MLFLOW_RUN_ID"):
        raise ValueError("Unset inherited MLFLOW_RUN_ID before recovery execution.")
    import mlflow

    if mlflow.active_run() is not None:
        raise ValueError("End the active MLflow run before recovery execution.")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execute_replay(
    evidence: SourceEvidence, plan: dict, output_root: Path, *, client: Any
) -> dict:
    """Run original seeded fit, then test its saved best validation checkpoint.

    This deliberately bypasses train.main's autolog/test-after-fit wrapper.
    MLflow and datamodule mutations happen only after all provenance gates pass.
    """
    _reject_inherited_context()
    expected = plan_replay(
        evidence,
        implementation_sha=plan["implementation_sha"],
        monitor=plan["monitor"] if plan["selection_changed"] else None,
        mode=plan["mode"] if plan["selection_changed"] else None,
    )
    if expected != plan:
        raise ValueError("Recovery plan changed; regenerate it from source evidence.")
    if parse_config_bytes(evidence.config_bytes) != evidence.config:
        raise ValueError(
            "Source config bytes and parsed training configuration disagree."
        )
    ensure_training_revision(plan["source_training_sha"])
    provenance = verify_replay_provenance(plan, evidence.config)
    # A repeated identity check closes the planning-to-execution window.
    source = client.get_run(plan["source_run_id"])
    if (
        source.info.status != "FINISHED"
        or source.info.lifecycle_stage != "active"
        or str(source.info.experiment_id) != plan["experiment_id"]
        or source.data.tags.get("mlflow.source.git.commit", plan["source_training_sha"])
        != plan["source_training_sha"]
    ):
        raise ValueError("Source identity/state changed since planning.")
    import hydra
    import torch
    from lightning.pytorch import seed_everything
    from omegaconf import OmegaConf

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f"recovery-{plan['source_run_id'][:8]}-", dir=output_root
        )
    )
    tags = {
        "recovery.source_run_id": plan["source_run_id"],
        "recovery.kind": "retrained",
        "recovery.source_training_sha": plan["source_training_sha"],
        "recovery.source_config_sha256": plan["source_config_sha256"],
        "recovery.implementation_sha": plan["implementation_sha"],
        "recovery.historical_test_metrics_revalidated": "false",
        "recovery.training_policy": plan["training_policy"],
    }
    run = client.create_run(
        experiment_id=plan["experiment_id"],
        tags=tags,
        run_name=f"checkpoint-recovery-{plan['source_run_id'][:8]}",
    )
    run_id = run.info.run_id
    if (
        run_id == plan["source_run_id"]
        or str(run.info.experiment_id) != plan["experiment_id"]
    ):
        raise ValueError("Server created an unexpected recovery run identity.")
    completed = False
    try:
        effective, changes = effective_config(evidence.config, plan, output, run_id)
        (output / "config-original.yaml").write_bytes(evidence.config_bytes)
        OmegaConf.save(OmegaConf.create(effective), output / "config-effective.yaml")
        _write_json(output / "config-allowlist-diff.json", {"changes": changes})
        _write_json(output / "plan.json", plan)
        _write_json(output / "provenance.json", provenance)
        for file in output.iterdir():
            client.log_artifact(run_id, str(file), "recovery")
        # Keep the standard source-run interface usable even when the caller
        # chooses recovery-only rather than replacing the original artifact.
        standard_config = output / "configs" / "config-resolved.yaml"
        standard_config.parent.mkdir()
        OmegaConf.save(OmegaConf.create(effective), standard_config)
        client.log_artifact(run_id, str(standard_config), "configs")
        cfg = OmegaConf.create(effective)
        logger = hydra.utils.instantiate(cfg.logger)
        _ = logger.experiment
        seed_everything(int(cfg.seed), workers=True)
        datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
        module = hydra.utils.instantiate(cfg.ml)
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        trainer.fit(module, datamodule=datamodule, ckpt_path=None)
        if trainer.interrupted or not trainer.state.finished:
            raise ValueError("Training did not finish normally; refusing publication.")
        checkpoint = trainer.checkpoint_callback
        if (
            checkpoint is None
            or not checkpoint.best_model_path
            or checkpoint.best_model_score is None
        ):
            raise ValueError("Fit produced no selected best checkpoint.")
        best = Path(checkpoint.best_model_path)
        if (
            best.parent.resolve() != (output / "checkpoints").resolve()
            or not best.is_file()
        ):
            raise ValueError("Best checkpoint escaped the recovery output directory.")
        payload = torch.load(best, map_location="cpu", weights_only=False)
        score = float(checkpoint.best_model_score.detach().cpu())
        matching_states = [
            state
            for state in payload.get("callbacks", {}).values()
            if isinstance(state, dict)
            and state.get("monitor") == plan["monitor"]
            and state.get("best_model_score") is not None
        ]
        if (
            len(matching_states) != 1
            or float(matching_states[0]["best_model_score"]) != score
        ):
            raise ValueError(
                "Saved checkpoint callback score disagrees with final selection."
            )
        best_hash = hashlib.sha256(best.read_bytes()).hexdigest()
        # Select on validation only, then test exactly that saved checkpoint.
        # The test result never controls checkpoint selection or publication.
        tests = trainer.test(
            module, datamodule=datamodule, ckpt_path=str(best), verbose=False
        )
        if trainer.interrupted or not trainer.state.finished or len(tests) != 1:
            raise ValueError("Best-checkpoint test evaluation did not finish normally.")
        if hashlib.sha256(best.read_bytes()).hexdigest() != best_hash:
            raise ValueError("Selected checkpoint changed during test evaluation.")
        # Force canonical best stem even if native filename versioning occurred.
        logger._log_checkpoint("best", str(best))
        manifest = result_manifest(
            plan,
            recovery_run_id=run_id,
            score=score,
            checkpoint_epoch=payload["epoch"],
            checkpoint_global_step=payload["global_step"],
            checkpoint_sha256=best_hash,
            checkpoint_size=best.stat().st_size,
            test_metrics=tests[0],
        )
        evaluation = output / "test-evaluation.json"
        _write_json(evaluation, manifest["test_evaluation"])
        client.log_artifact(run_id, str(evaluation), "recovery")
        with tempfile.TemporaryDirectory(prefix="recovery-readback-") as directory:
            remote = Path(client.download_artifacts(run_id, CHECKPOINT, directory))
            if (
                remote.stat().st_size != manifest["checkpoint_size"]
                or hashlib.sha256(remote.read_bytes()).hexdigest()
                != manifest["checkpoint_sha256"]
            ):
                raise ValueError(
                    "Recovery checkpoint artifact readback does not match final local best."
                )
            model_path = Path(client.download_artifacts(run_id, MLMODEL, directory))
            import yaml

            model = yaml.safe_load(model_path.read_bytes())
            if (
                model.get("run_id") != run_id
                or model.get("model_size_bytes") != manifest["checkpoint_size"]
                or model.get("flavors", {}).get("pytorch", {}).get("model_data")
                != "checkpoint.ckpt"
            ):
                raise ValueError(
                    "Recovery MLmodel readback does not bind the checkpoint bytes/run."
                )
        manifest["provenance"] = provenance
        manifest["output_directory"] = str(output)
        _write_json(output / "recovery-manifest.json", manifest)
        client.log_artifact(run_id, str(output / "recovery-manifest.json"), "recovery")
        client.set_tag(
            run_id,
            "recovery.historical_target_reached",
            str(manifest["historical_target_reached"]).lower(),
        )
        client.set_terminated(run_id, status="FINISHED")
        completed = True
        return manifest
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    finally:
        import mlflow

        if (
            mlflow.active_run() is not None
            and mlflow.active_run().info.run_id == run_id
        ):
            mlflow.end_run(status="FINISHED" if completed else "FAILED")


def recover_and_replace(
    evidence: SourceEvidence,
    plan: dict,
    output_root: Path,
    *,
    client: Any,
    replace_original: bool = False,
    expected_old_sha256: str | None = None,
) -> dict:
    """Pin original bytes; publish after successful original-protocol fit/test.

    Publication checks this pin again after training and after verified backups.
    This is a single-writer guard, not an atomic object-store transaction.
    """
    _reject_inherited_context()
    from ml.checkpoint_publication import publish_replacement, read_checkpoint_hash

    old_sha = None
    if replace_original:
        old_sha = read_checkpoint_hash(client, plan["source_run_id"])
        if expected_old_sha256 is not None and old_sha != expected_old_sha256:
            raise ValueError("Original checkpoint differs from the supplied SHA256.")
    manifest = execute_replay(evidence, plan, output_root, client=client)
    if not replace_original:
        return manifest
    receipt = publish_replacement(client, manifest, old_sha)
    output = Path(manifest["output_directory"]) / "publication-receipt.json"
    try:
        _write_json(output, receipt)
        client.log_artifact(manifest["recovery_run_id"], str(output), "recovery")
    except Exception as exc:
        # Canonical publication already succeeded and has its own source-run
        # journal. Report this optional extra receipt-copy failure accurately.
        receipt["recovery_receipt_persistence_error"] = type(exc).__name__
    return {"recovery": manifest, "publication": receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="MLflow run URL or safe run ID")
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="Explicit credentials-free HTTP(S) endpoint",
    )
    parser.add_argument(
        "--implementation-sha",
        required=True,
        help="Exact reviewed implementation Git SHA",
    )
    parser.add_argument(
        "--training-sha", help="Exact historical SHA only when source tag is absent"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("checkpoint-recoveries")
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--execute",
        action="store_true",
        help="Train/validate, then test best checkpoint in a fresh linked recovery run",
    )
    actions.add_argument(
        "--publish",
        action="store_true",
        help="Separate guarded replacement; never trains",
    )
    parser.add_argument("--replace-original", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-old-sha256")
    parser.add_argument(
        "--monitor", help="Explicit alternative valid/... checkpoint objective"
    )
    parser.add_argument("--mode", choices=("min", "max"))
    parser.add_argument("--max-history-records", type=int, default=200_000)
    parser.add_argument("--max-history-bytes", type=int, default=32 * 1024 * 1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from ml.checkpoint_publication import PublicationError

    try:
        identity = parse_source(args.source, args.tracking_uri)
        _sha(args.implementation_sha)
        if (args.monitor is None) != (args.mode is None):
            raise ValueError("--monitor and --mode must be supplied together.")
        if args.publish:
            if (
                not args.replace_original
                or args.manifest is None
                or not args.expected_old_sha256
            ):
                raise ValueError(
                    "Publication requires --replace-original, --manifest and --expected-old-sha256."
                )
            if args.monitor or args.training_sha:
                raise ValueError(
                    "Publish uses manifest provenance; selection/training overrides are forbidden."
                )
            if args.manifest.stat().st_size > 1024 * 1024:
                raise ValueError("Publication manifest exceeds 1 MiB.")
            manifest = json.loads(args.manifest.read_text())
            if (
                manifest.get("source_run_id") != identity.run_id
                or normalize_tracking_uri(manifest.get("tracking_uri", ""))
                != identity.tracking_uri
                or manifest.get("implementation_sha") != args.implementation_sha
                or (
                    identity.experiment_id is not None
                    and manifest.get("experiment_id") != identity.experiment_id
                )
            ):
                raise ValueError(
                    "Publish source/experiment/tracking/implementation identity differs from manifest."
                )
            from mlflow import MlflowClient

            from ml.checkpoint_publication import publish_replacement

            client = MlflowClient(tracking_uri=identity.tracking_uri)
            receipt = publish_replacement(client, manifest, args.expected_old_sha256)
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.manifest or (
            (args.replace_original or args.expected_old_sha256) and not args.execute
        ):
            raise ValueError(
                "Replacement requires explicit --execute or --publish; --manifest is publish-only."
            )
        if args.expected_old_sha256 and (
            not args.replace_original
            or not re.fullmatch(r"[0-9a-f]{64}", args.expected_old_sha256)
        ):
            raise ValueError(
                "Expected old SHA256 requires --replace-original and 64 lowercase hex characters."
            )
        limits = MetadataLimits(
            max_records=args.max_history_records, history_bytes=args.max_history_bytes
        )
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=identity.tracking_uri)
        source = read_source(
            client,
            identity,
            limits=limits,
            training_sha=args.training_sha,
            monitor=args.monitor,
            mode=args.mode,
        )
        plan = plan_replay(
            source,
            implementation_sha=args.implementation_sha,
            monitor=args.monitor,
            mode=args.mode,
        )
        result = (
            recover_and_replace(
                source,
                plan,
                args.output_root,
                client=client,
                replace_original=args.replace_original,
                expected_old_sha256=args.expected_old_sha256,
            )
            if args.execute
            else plan
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except PublicationError as exc:
        print(json.dumps(exc.receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
