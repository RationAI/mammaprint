"""Guarded publication of retrained checkpoints through MLflow's public client.

The caller must explicitly authorize publication and serialize writers per source
run. MLflow object writes are not atomic and provide no compare-and-swap. A phase
journal and immutable operation artifacts preserve evidence for manual recovery;
this module never deletes artifacts or automatically rolls back a partial write.
"""

import ast
import hashlib
import json
import math
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit


CHECKPOINT_PATH = "checkpoints/best/checkpoint.ckpt"
MODEL_PATH = "checkpoints/best/MLmodel"
CONFIG_PATH = "configs/config-resolved.yaml"
CHECKPOINT_LIMIT = 512 * 1024 * 1024
METADATA_LIMIT = 1024 * 1024


class PublicationError(RuntimeError):
    """A failed operation with its retained local journal and observed receipt."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt
        self.journal_path = receipt["journal_path"]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def _finite(value, name):
    _require(not isinstance(value, bool), f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _hex(value, length, name):
    _require(
        isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value),
        f"Invalid {name}",
    )


def _tracking_uri(value):
    _require(
        isinstance(value, str) and value and not any(ord(c) <= 32 for c in value),
        "Invalid tracking URI",
    )
    parsed = urlsplit(value)
    _require(
        parsed.scheme in ("http", "https")
        and parsed.hostname
        and not (parsed.username or parsed.password or parsed.query or parsed.fragment),
        "Tracking URI must be an explicit credentials-free HTTP(S) endpoint",
    )
    _require(
        "%" not in parsed.netloc
        and "\\" not in value
        and not any(p in (".", "..") for p in parsed.path.split("/")),
        "Invalid tracking URI",
    )
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid tracking URI port") from exc
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


def _validate_manifest(manifest, expected_old):
    _require(isinstance(manifest, dict), "Recovery manifest must be an object")
    required = {
        "format_version",
        "status",
        "kind",
        "tracking_uri",
        "experiment_id",
        "source_run_id",
        "recovery_run_id",
        "source_training_sha",
        "implementation_sha",
        "source_config_sha256",
        "monitor",
        "mode",
        "observed_best_score",
        "checkpoint_artifact_path",
        "checkpoint_sha256",
        "checkpoint_size",
        "checkpoint_epoch",
        "checkpoint_global_step",
        "model_artifact_path",
        "historical_test_metrics_revalidated",
    }
    _require(
        required <= manifest.keys(),
        f"Missing recovery manifest fields: {sorted(required - manifest.keys())}",
    )
    _require(
        type(manifest["format_version"]) is int
        and manifest["format_version"] in (1, 2),
        "Unsupported manifest version",
    )
    _require(
        manifest["status"] == "completed" and manifest["kind"] == "retrained",
        "Only completed retraining can be published",
    )
    _require(
        manifest["historical_test_metrics_revalidated"] is False,
        "Historical test metrics cannot be reattributed",
    )
    for name in ("source_run_id", "recovery_run_id"):
        _hex(manifest[name], 32, name)
    _require(
        manifest["source_run_id"] != manifest["recovery_run_id"],
        "Recovery must use a distinct run",
    )
    for name in ("source_training_sha", "implementation_sha"):
        _hex(manifest[name], 40, name)
    for name in ("source_config_sha256", "checkpoint_sha256"):
        _hex(manifest[name], 64, name)
    _hex(expected_old, 64, "expected old checkpoint SHA256")
    _tracking_uri(manifest["tracking_uri"])
    _require(
        isinstance(manifest["experiment_id"], str)
        and manifest["experiment_id"].isdigit(),
        "Invalid experiment ID",
    )
    monitor = manifest["monitor"]
    _require(
        isinstance(monitor, str)
        and monitor.startswith(("valid/", "val/"))
        and not any(ord(c) < 32 for c in monitor),
        "Publication requires a validation monitor",
    )
    _require(manifest["mode"] in ("min", "max"), "Invalid checkpoint mode")
    for name in ("checkpoint_epoch", "checkpoint_global_step", "checkpoint_size"):
        _integer(manifest[name], name)
    _require(manifest["checkpoint_global_step"] > 0, "Checkpoint has no training steps")
    observed = _finite(manifest["observed_best_score"], "observed score")
    if manifest["format_version"] == 1:
        # Previously completed bounded recoveries retain their original contract.
        _require(
            {"target_score", "stop_epoch", "target_reached"} <= manifest.keys(),
            "Missing legacy recovery target fields",
        )
        _require(manifest["target_reached"] is True, "Recovery target was not reached")
        target = _finite(manifest["target_score"], "target score")
        reached = (
            observed <= target if manifest["mode"] == "min" else observed >= target
        )
        _require(reached, "Recovery score did not reach its directional target")
        _integer(manifest["stop_epoch"], "stop_epoch")
        _require(
            manifest["checkpoint_epoch"] <= manifest["stop_epoch"],
            "Checkpoint exceeds replay epoch budget",
        )
    else:
        _require(
            {
                "training_policy",
                "trainer_max_epochs",
                "trainer_max_steps",
                "historical_best_score",
                "historical_target_reached",
                "test_evaluation",
            }
            <= manifest.keys(),
            "Missing original-protocol recovery fields",
        )
        _require(
            manifest["training_policy"] == "original_horizon_and_early_stopping",
            "Unsupported retraining policy",
        )
        for name in ("trainer_max_epochs", "trainer_max_steps"):
            value = manifest[name]
            _require(
                type(value) is int and (value == -1 or value > 0),
                f"Invalid original training limit: {name}",
            )
        _require(
            manifest["trainer_max_epochs"] != -1 or manifest["trainer_max_steps"] != -1,
            "Original training needs a finite limit",
        )
        if manifest["trainer_max_epochs"] != -1:
            _require(
                type(manifest["checkpoint_epoch"]) is int
                and 0 <= manifest["checkpoint_epoch"] < manifest["trainer_max_epochs"],
                "Checkpoint exceeds original epoch limit",
            )
        if manifest["trainer_max_steps"] != -1:
            _require(
                type(manifest["checkpoint_global_step"]) is int
                and manifest["checkpoint_global_step"] <= manifest["trainer_max_steps"],
                "Checkpoint exceeds original step limit",
            )
        historical = _finite(manifest["historical_best_score"], "historical score")
        reached = (
            observed <= historical
            if manifest["mode"] == "min"
            else observed >= historical
        )
        _require(
            type(manifest["historical_target_reached"]) is bool
            and manifest["historical_target_reached"] == reached,
            "Historical comparison is inconsistent",
        )
        evaluation = manifest["test_evaluation"]
        _require(isinstance(evaluation, dict), "Missing test evaluation")
        for key in ("checkpoint_sha256", "checkpoint_epoch", "checkpoint_global_step"):
            _require(
                evaluation.get(key) == manifest[key],
                "Test evaluation checkpoint mismatch",
            )
        metrics = evaluation.get("metrics")
        _require(isinstance(metrics, dict) and bool(metrics), "Missing test metrics")
        for key, value in metrics.items():
            _require(
                isinstance(key, str) and key.startswith("test/"),
                "Invalid test metric key",
            )
            _finite(value, "test metric")
    _require(
        0 < manifest["checkpoint_size"] <= CHECKPOINT_LIMIT,
        "Checkpoint size exceeds allowed bounds",
    )
    _require(
        manifest["checkpoint_artifact_path"] == CHECKPOINT_PATH,
        "Unexpected checkpoint artifact path",
    )
    _require(
        manifest["model_artifact_path"] == MODEL_PATH,
        "Unexpected MLmodel artifact path",
    )


def _read(client, run_id, path, limit, scratch):
    """Check SDK file metadata before download, then enforce actual size bounds."""
    parent = str(PurePosixPath(path).parent)
    if parent == ".":
        parent = ""
    entries = [
        entry for entry in client.list_artifacts(run_id, parent) if entry.path == path
    ]
    _require(
        len(entries) == 1 and not entries[0].is_dir,
        f"Missing or ambiguous artifact: {path}",
    )
    size = getattr(entries[0], "file_size", None)
    _require(
        type(size) is int and 0 < size <= limit,
        f"Artifact size unavailable or exceeds limit: {path}",
    )
    download_dir = Path(tempfile.mkdtemp(prefix="read-", dir=scratch))
    local = Path(client.download_artifacts(run_id, path, str(download_dir)))
    _require(
        local.is_file() and not local.is_symlink(),
        f"Artifact is not a regular file: {path}",
    )
    actual = local.stat().st_size
    _require(
        actual == size and 0 < actual <= limit,
        f"Artifact downloaded size mismatch: {path}",
    )
    return local.read_bytes()


def read_checkpoint_hash(client, run_id: str) -> str:
    """Read-only pin used before retraining; publication verifies it again."""
    _hex(run_id, 32, "source run ID")
    with tempfile.TemporaryDirectory(prefix="checkpoint-original-pin-") as directory:
        return _sha(
            _read(client, run_id, CHECKPOINT_PATH, CHECKPOINT_LIMIT, Path(directory))
        )


def _verify_read(client, run_id, path, expected, limit, scratch):
    actual = _read(client, run_id, path, limit, scratch)
    _require(
        _sha(actual) == _sha(expected), f"Artifact readback SHA256 mismatch: {path}"
    )
    return actual


def _run_identity(client, manifest):
    uri = getattr(client, "tracking_uri", None)
    _require(
        _tracking_uri(uri) == _tracking_uri(manifest["tracking_uri"]),
        "Client tracking URI differs from manifest",
    )
    runs = {}
    for name in ("source_run_id", "recovery_run_id"):
        run_id = manifest[name]
        run = client.get_run(run_id)
        _require(run.info.run_id == run_id, f"Run identity mismatch: {name}")
        _require(
            str(run.info.experiment_id) == manifest["experiment_id"],
            "Run experiment mismatch",
        )
        _require(
            run.info.status == "FINISHED" and run.info.lifecycle_stage == "active",
            "Both runs must be active and FINISHED",
        )
        runs[name] = run
    source_tags = runs["source_run_id"].data.tags
    _require(
        source_tags.get("mlflow.source.git.commit") == manifest["source_training_sha"],
        "Source Git SHA mismatch",
    )
    recovery_tags = runs["recovery_run_id"].data.tags
    bindings = {
        "recovery.source_run_id": manifest["source_run_id"],
        "recovery.source_training_sha": manifest["source_training_sha"],
        "recovery.source_config_sha256": manifest["source_config_sha256"],
        "recovery.kind": "retrained",
    }
    for key, expected in bindings.items():
        # Older compatible runs may lack a tag; any present binding must agree.
        if key in recovery_tags:
            _require(
                recovery_tags[key] == expected, f"Recovery run binding mismatch: {key}"
            )


def _model(data, run_id, size):
    import yaml

    try:
        model = yaml.safe_load(data)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ValueError("Invalid MLmodel YAML") from exc
    _require(isinstance(model, dict), "MLmodel must be a mapping")
    flavors = model.get("flavors")
    _require(isinstance(flavors, dict), "MLmodel flavors must be a mapping")
    flavor = flavors.get("pytorch", {})
    _require(
        isinstance(flavor, dict) and flavor.get("model_data") == "checkpoint.ckpt",
        "Unsupported native checkpoint MLmodel flavor",
    )
    _require(
        isinstance(flavor.get("pytorch_version"), str) and flavor["pytorch_version"],
        "MLmodel lacks PyTorch version",
    )
    _require(model.get("run_id") == run_id, "MLmodel run binding mismatch")
    _require(
        type(model.get("model_size_bytes")) is int
        and model["model_size_bytes"] == size,
        "MLmodel size mismatch",
    )
    return model


def _checkpoint(data, manifest, scratch):
    """Use restricted tensor/state loading; never fall back to arbitrary pickle."""
    import torch

    path = Path(scratch) / "trusted-recovery.ckpt"
    path.write_bytes(data)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Checkpoint cannot be loaded with weights_only=True") from exc
    _require(isinstance(state, dict), "Checkpoint must be a mapping")
    for key, expected_name in (
        ("epoch", "checkpoint_epoch"),
        ("global_step", "checkpoint_global_step"),
    ):
        _require(
            type(state.get(key)) is int and state[key] == manifest[expected_name],
            f"Checkpoint {key} mismatch",
        )
    callbacks = state.get("callbacks")
    _require(isinstance(callbacks, dict), "Checkpoint lacks ModelCheckpoint state")
    matches = []
    for key, callback in callbacks.items():
        if (
            not isinstance(key, str)
            or not key.startswith("ModelCheckpoint")
            or not isinstance(callback, dict)
        ):
            continue
        try:
            options = ast.literal_eval(key[len("ModelCheckpoint") :])
        except (SyntaxError, ValueError):
            options = {}
        if not isinstance(options, dict):
            continue
        for name in ("monitor", "mode"):
            if name in options and name in callback:
                _require(
                    callback[name] == options[name],
                    f"Conflicting checkpoint callback {name}",
                )
        monitor = callback.get("monitor", options.get("monitor"))
        mode = callback.get("mode", options.get("mode"))
        if monitor == manifest["monitor"] and mode == manifest["mode"]:
            matches.append(callback)
    _require(len(matches) == 1, "Checkpoint selection state is missing or ambiguous")
    score = _finite(matches[0].get("best_model_score"), "checkpoint best score")
    _require(
        score == _finite(manifest["observed_best_score"], "observed score"),
        "Checkpoint best score mismatch",
    )
    binding = state.get("recovery", {})
    _require(isinstance(binding, dict), "Invalid checkpoint recovery binding")
    for key in ("source_run_id", "source_training_sha", "source_config_sha256"):
        if key in binding:
            _require(
                binding[key] == manifest[key],
                f"Checkpoint source binding mismatch: {key}",
            )


def _upload(client, run_id, path, data, scratch):
    local_dir = Path(tempfile.mkdtemp(prefix="write-", dir=scratch))
    local = local_dir / PurePosixPath(path).name
    local.write_bytes(data)
    client.log_artifact(run_id, str(local), str(PurePosixPath(path).parent))


def _json(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_replacement(client, manifest: dict, expected_old_sha256: str) -> dict:
    """Guard, back up, stage, replace and read-verify two canonical artifacts.

    Validation errors before the first mutation raise ``ValueError``. Once an
    operation starts, failures raise ``PublicationError`` with a retained local
    journal path, remote namespace and partial-state receipt. The caller must
    configure the client for ``manifest['tracking_uri']`` and enforce one writer.
    """
    _validate_manifest(manifest, expected_old_sha256)
    _run_identity(client, manifest)
    source = manifest["source_run_id"]
    recovery = manifest["recovery_run_id"]
    with tempfile.TemporaryDirectory(
        prefix="checkpoint-publication-validation-"
    ) as scratch:
        config = _read(client, source, CONFIG_PATH, METADATA_LIMIT, scratch)
        _require(
            _sha(config) == manifest["source_config_sha256"],
            "Source config SHA256 mismatch",
        )
        if manifest["format_version"] == 2:
            import yaml

            original = yaml.safe_load(config)
            _require(
                isinstance(original, dict)
                and isinstance(original.get("trainer"), dict),
                "Source config lacks original trainer settings",
            )
            trainer = original.get("trainer", {})
            epochs, steps = (
                trainer.get("max_epochs", 1000),
                trainer.get("max_steps", -1),
            )
            if epochs is None:
                epochs = 1000 if steps == -1 else -1
            _require(
                epochs == manifest["trainer_max_epochs"]
                and steps == manifest["trainer_max_steps"],
                "Original training limits differ from source config",
            )
            evaluation = _read(
                client,
                recovery,
                "recovery/test-evaluation.json",
                METADATA_LIMIT,
                scratch,
            )
            _require(
                json.loads(evaluation) == manifest["test_evaluation"],
                "Test evaluation artifact differs from recovery manifest",
            )
        old = _read(client, source, CHECKPOINT_PATH, CHECKPOINT_LIMIT, scratch)
        _require(
            _sha(old) == expected_old_sha256, "Expected old checkpoint SHA256 mismatch"
        )
        old_model = _read(client, source, MODEL_PATH, METADATA_LIMIT, scratch)
        _model(old_model, source, len(old))
        new = _read(client, recovery, CHECKPOINT_PATH, CHECKPOINT_LIMIT, scratch)
        _require(
            len(new) == manifest["checkpoint_size"]
            and _sha(new) == manifest["checkpoint_sha256"],
            "Recovery checkpoint SHA256/size mismatch",
        )
        _checkpoint(new, manifest, scratch)
        model_bytes = _read(client, recovery, MODEL_PATH, METADATA_LIMIT, scratch)
        model = _model(model_bytes, recovery, len(new))
        model["run_id"] = source
        model["model_size_bytes"] = len(new)
        import yaml

        replacement_model = yaml.safe_dump(model, sort_keys=False).encode()
        _require(
            len(replacement_model) <= METADATA_LIMIT,
            "Replacement MLmodel exceeds size limit",
        )
        _model(replacement_model, source, len(new))
        namespace = f"checkpoint_repairs/{uuid.uuid4()}"
        existing = client.list_artifacts(source, "checkpoint_repairs")
        _require(
            not any(entry.path == namespace for entry in existing),
            "Operation namespace already exists; refusing overwrite",
        )
        _require(
            not client.list_artifacts(source, namespace),
            "Operation namespace already exists; refusing overwrite",
        )
        # The local journal outlives temporary validation files, including failures.
        journal_dir = Path(tempfile.mkdtemp(prefix="checkpoint-publication-journal-"))
        journal_path = journal_dir / "journal.json"
        receipt = {
            "format_version": 1,
            "status": "in_progress",
            "phase": "validated",
            "source_run_id": source,
            "recovery_run_id": recovery,
            "tracking_uri": manifest["tracking_uri"],
            "experiment_id": manifest["experiment_id"],
            "namespace": namespace,
            "journal_path": str(journal_path),
            "old_checkpoint_sha256": expected_old_sha256,
            "new_checkpoint_sha256": _sha(new),
            "old_mlmodel_sha256": _sha(old_model),
            "new_mlmodel_sha256": _sha(replacement_model),
            "canonical_write_attempted": False,
            "historical_test_metrics_revalidated": False,
            "atomic_compare_and_swap": False,
            "requires_single_writer": True,
            "phases": [],
        }

        def journal(phase):
            receipt["phase"] = phase
            receipt["phases"].append(phase)
            journal_path.write_bytes(_json(receipt))
            _upload(
                client, source, f"{namespace}/journal.json", _json(receipt), scratch
            )

        try:
            journal("backing_up")
            operation = _json({"manifest": manifest, "publication": receipt.copy()})
            for suffix, data, limit in (
                ("backup/checkpoint.ckpt", old, CHECKPOINT_LIMIT),
                ("backup/MLmodel", old_model, METADATA_LIMIT),
                ("operation.json", operation, METADATA_LIMIT),
            ):
                _upload(client, source, f"{namespace}/{suffix}", data, scratch)
                _verify_read(
                    client, source, f"{namespace}/{suffix}", data, limit, scratch
                )
            journal("staging")
            for suffix, data, limit in (
                ("staged/checkpoint.ckpt", new, CHECKPOINT_LIMIT),
                ("staged/MLmodel", replacement_model, METADATA_LIMIT),
            ):
                _upload(client, source, f"{namespace}/{suffix}", data, scratch)
                _verify_read(
                    client, source, f"{namespace}/{suffix}", data, limit, scratch
                )
            journal("rechecking_original")
            _run_identity(client, manifest)
            _verify_read(client, source, MODEL_PATH, old_model, METADATA_LIMIT, scratch)
            _verify_read(
                client, source, CHECKPOINT_PATH, old, CHECKPOINT_LIMIT, scratch
            )
            # No mutation between the expected-old read and the checkpoint write.
            receipt["canonical_write_attempted"] = True
            receipt["phase"] = "replacing_checkpoint"
            receipt["phases"].append("replacing_checkpoint")
            journal_path.write_bytes(_json(receipt))
            _upload(client, source, CHECKPOINT_PATH, new, scratch)
            journal("replacing_mlmodel")
            _upload(client, source, MODEL_PATH, replacement_model, scratch)
            journal("verifying_canonical")
            verified = _verify_read(
                client, source, CHECKPOINT_PATH, new, CHECKPOINT_LIMIT, scratch
            )
            _checkpoint(verified, manifest, scratch)
            verified_model = _verify_read(
                client, source, MODEL_PATH, replacement_model, METADATA_LIMIT, scratch
            )
            _model(verified_model, source, len(new))
            journal("recording_provenance")
            tags = {
                "recovery.publication_status": "completed",
                "recovery.kind": "retrained",
                "recovery.recovery_run_id": recovery,
                "recovery.publication_namespace": namespace,
                "recovery.old_checkpoint_sha256": expected_old_sha256,
                "recovery.new_checkpoint_sha256": _sha(new),
                "recovery.historical_test_metrics_revalidated": "false",
            }
            if manifest["format_version"] == 2:
                tags.update(
                    {
                        "recovery.training_policy": manifest["training_policy"],
                        "recovery.test_metrics_run_id": recovery,
                        "recovery.test_evaluation_artifact": "recovery/test-evaluation.json",
                        "recovery.historical_target_reached": str(
                            manifest["historical_target_reached"]
                        ).lower(),
                    }
                )
            # Success marker is last; a failed tag write never reports completion.
            for key, value in tags.items():
                if key != "recovery.publication_status":
                    client.set_tag(source, key, value)
            receipt["status"] = "completed"
            journal("completed")
            client.set_tag(source, "recovery.publication_status", "completed")
            return receipt
        except Exception as exc:
            receipt["failed_phase"] = receipt["phase"]
            receipt["status"] = (
                "partial" if receipt["canonical_write_attempted"] else "failed"
            )
            receipt["error_type"] = type(exc).__name__
            receipt["phase"] = "failed"
            errors = receipt.setdefault("journal_persistence_errors", {})
            # Failure reporting must survive a full/unwritable local filesystem,
            # especially after a canonical artifact has already changed.
            try:
                journal_path.write_bytes(_json(receipt))
            except Exception as journal_exc:
                errors["local"] = type(journal_exc).__name__
            try:
                _upload(
                    client, source, f"{namespace}/journal.json", _json(receipt), scratch
                )
            except Exception as journal_exc:
                errors["remote"] = type(journal_exc).__name__
            try:
                client.set_tag(source, "recovery.publication_status", receipt["status"])
                client.set_tag(
                    source, "recovery.publication_failed_phase", receipt["failed_phase"]
                )
            except Exception as tag_exc:
                errors["tags"] = type(tag_exc).__name__
            raise PublicationError(
                f"Checkpoint publication {receipt['status']} during {receipt['failed_phase']}; journal: {journal_path}",
                receipt,
            ) from exc
