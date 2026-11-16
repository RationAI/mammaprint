"""Guarded publication against a local, byte-preserving MLflow artifact fake."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from ml.checkpoint_publication import PublicationError, publish_replacement


SOURCE = "a" * 32
RECOVERY = "b" * 32
OLD_SHA = "c" * 40
IMPLEMENTATION_SHA = "d" * 40
CHECKPOINT = "checkpoints/best/checkpoint.ckpt"
MODEL = "checkpoints/best/MLmodel"
NAMESPACE = "checkpoint_repairs/fixed-operation"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ArtifactClient:
    """The only substituted boundary is remote transport, including real files."""

    tracking_uri = "https://mlflow.example.test"

    def __init__(self, root):
        self.root = root
        self.writes = []
        self.tags = []
        self.reads = []
        self.fail_write = None
        self.corrupt_read = None
        self.before_read = None
        self.runs = {}

    def path(self, run_id, path):
        return self.root / run_id / path

    def get_run(self, run_id):
        return self.runs[run_id]

    def list_artifacts(self, run_id, path):
        parent = self.path(run_id, path)
        if not parent.is_dir():
            return []
        return [
            SimpleNamespace(
                path=f"{path}/{child.name}" if path else child.name,
                is_dir=child.is_dir(),
                file_size=child.stat().st_size if child.is_file() else None,
            )
            for child in parent.iterdir()
        ]

    def download_artifacts(self, run_id, path, dst_path):
        self.reads.append((run_id, path))
        if self.before_read:
            self.before_read(run_id, path)
        target = Path(dst_path) / Path(path).name
        shutil.copyfile(self.path(run_id, path), target)
        if self.corrupt_read and self.corrupt_read(run_id, path):
            target.write_bytes(b"damaged")
        return str(target)

    def log_artifact(self, run_id, local_path, artifact_path):
        path = f"{artifact_path}/{Path(local_path).name}"
        if self.fail_write and self.fail_write(run_id, path):
            raise OSError("simulated upload failure")
        target = self.path(run_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)
        self.writes.append((run_id, path))

    def set_tag(self, run_id, key, value):
        self.tags.append((run_id, key, str(value)))
        self.runs[run_id].data.tags[key] = str(value)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = ArtifactClient(self.root)
        config = b"seed: 1\nml: native\n"
        self.manifest = {
            "format_version": 1,
            "status": "completed",
            "kind": "retrained",
            "tracking_uri": self.client.tracking_uri,
            "experiment_id": "9",
            "source_run_id": SOURCE,
            "recovery_run_id": RECOVERY,
            "source_training_sha": OLD_SHA,
            "implementation_sha": IMPLEMENTATION_SHA,
            "source_config_sha256": sha(config),
            "monitor": "valid/loss",
            "mode": "min",
            "target_score": 0.4,
            "stop_epoch": 3,
            "observed_best_score": 0.3,
            "target_reached": True,
            "checkpoint_artifact_path": CHECKPOINT,
            "checkpoint_epoch": 2,
            "checkpoint_global_step": 8,
            "model_artifact_path": MODEL,
            "historical_test_metrics_revalidated": False,
        }
        self.old = b"old original checkpoint"
        old_model = self.model_bytes(SOURCE, len(self.old))
        self.put(SOURCE, CHECKPOINT, self.old)
        self.put(SOURCE, MODEL, old_model)
        self.put(SOURCE, "configs/config-resolved.yaml", config)
        self.put(SOURCE, "console.log", b"old console")
        source_tags = {"mlflow.source.git.commit": OLD_SHA, "user.tag": "keep"}
        recovery_tags = {
            "recovery.source_run_id": SOURCE,
            "recovery.source_training_sha": OLD_SHA,
            "recovery.source_config_sha256": sha(config),
            "recovery.kind": "retrained",
        }
        for run_id, tags in ((SOURCE, source_tags), (RECOVERY, recovery_tags)):
            self.client.runs[run_id] = SimpleNamespace(
                info=SimpleNamespace(
                    run_id=run_id,
                    experiment_id="9",
                    status="FINISHED",
                    lifecycle_stage="active",
                ),
                data=SimpleNamespace(tags=tags, metrics={"test/f1": 0.9}),
            )
        path = self.root / "new.ckpt"
        torch.save(
            {
                "epoch": 2,
                "global_step": 8,
                "state_dict": {"weight": torch.tensor([1.0])},
                "callbacks": {
                    "ModelCheckpoint{'monitor': 'valid/loss', 'mode': 'min'}": {
                        "monitor": "valid/loss",
                        "best_model_score": torch.tensor(0.3, dtype=torch.float64),
                    }
                },
            },
            path,
        )
        self.new = path.read_bytes()
        self.manifest.update(
            checkpoint_sha256=sha(self.new), checkpoint_size=len(self.new)
        )
        self.put(RECOVERY, CHECKPOINT, self.new)
        self.put(RECOVERY, MODEL, self.model_bytes(RECOVERY, len(self.new)))
        self.uuid_patch = patch(
            "ml.checkpoint_publication.uuid.uuid4", return_value="fixed-operation"
        )
        self.uuid_patch.start()
        self.addCleanup(self.uuid_patch.stop)

    @staticmethod
    def model_bytes(run_id, size):
        return yaml.safe_dump(
            {
                "run_id": run_id,
                "model_size_bytes": size,
                "flavors": {
                    "pytorch": {
                        "model_data": "checkpoint.ckpt",
                        "pytorch_version": "2.8.0",
                    }
                },
                "native_config": {"preserve": True},
            }
        ).encode()

    def put(self, run_id, path, data):
        target = self.client.path(run_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def publish(self):
        return publish_replacement(self.client, self.manifest, sha(self.old))

    def assert_no_canonical_writes(self):
        self.assertFalse(
            any(path in (CHECKPOINT, MODEL) for _, path in self.client.writes)
        )

    def test_success_preserves_history_and_only_replaces_two_canonical_artifacts(self):
        old_tags = copy.deepcopy(self.client.runs[SOURCE].data.tags)
        old_metrics = copy.deepcopy(self.client.runs[SOURCE].data.metrics)
        receipt = self.publish()
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(self.client.path(SOURCE, CHECKPOINT).read_bytes(), self.new)
        model = yaml.safe_load(self.client.path(SOURCE, MODEL).read_bytes())
        self.assertEqual(model["run_id"], SOURCE)
        self.assertEqual(model["model_size_bytes"], len(self.new))
        self.assertEqual(model["native_config"], {"preserve": True})
        self.assertEqual(
            self.client.path(SOURCE, "configs/config-resolved.yaml").read_bytes(),
            b"seed: 1\nml: native\n",
        )
        self.assertEqual(
            self.client.path(SOURCE, "console.log").read_bytes(), b"old console"
        )
        self.assertEqual(self.client.runs[SOURCE].data.metrics, old_metrics)
        for key, value in old_tags.items():
            self.assertEqual(self.client.runs[SOURCE].data.tags[key], value)
        self.assertEqual(
            self.client.runs[SOURCE].data.tags[
                "recovery.historical_test_metrics_revalidated"
            ],
            "false",
        )
        self.assertEqual(
            self.client.path(
                SOURCE, f"{NAMESPACE}/backup/checkpoint.ckpt"
            ).read_bytes(),
            self.old,
        )
        self.assertEqual(
            json.loads(Path(receipt["journal_path"]).read_text())["phase"], "completed"
        )
        canonical_index = next(
            i for i, (_, path) in enumerate(self.client.writes) if path == CHECKPOINT
        )
        before = self.client.writes[:canonical_index]
        for path in (
            "backup/checkpoint.ckpt",
            "backup/MLmodel",
            "operation.json",
            "staged/checkpoint.ckpt",
            "staged/MLmodel",
        ):
            self.assertIn((SOURCE, f"{NAMESPACE}/{path}"), before)
            self.assertIn((SOURCE, f"{NAMESPACE}/{path}"), self.client.reads)

    def test_manifest_guards_fail_without_mutation(self):
        changes = [
            {"status": "target_missed"},
            {"target_reached": False},
            {"observed_best_score": 0.5},
            {"observed_best_score": float("nan")},
            {"target_score": float("inf")},
            {"kind": "reconstructed"},
            {"historical_test_metrics_revalidated": True},
            {"source_config_sha256": "0" * 64},
            {"checkpoint_sha256": "0" * 64},
            {"checkpoint_size": 1},
            {"checkpoint_epoch": 4},
            {"checkpoint_global_step": 9},
            {"source_training_sha": "e" * 40},
            {"tracking_uri": "https://wrong.test"},
            {"monitor": "test/loss"},
            {"checkpoint_artifact_path": "other.ckpt"},
        ]
        original = copy.deepcopy(self.manifest)
        for change in changes:
            with self.subTest(change=change):
                self.manifest = original | change
                with self.assertRaises((ValueError, PublicationError)):
                    self.publish()
                self.assertEqual(self.client.writes, [])
                self.assertEqual(self.client.tags, [])
        self.manifest = original

    def test_identity_status_and_tag_mismatch_fail_before_mutation(self):
        cases = [
            (SOURCE, "experiment_id", "7"),
            (RECOVERY, "experiment_id", "7"),
            (SOURCE, "status", "RUNNING"),
            (RECOVERY, "status", "FAILED"),
            (SOURCE, "lifecycle_stage", "deleted"),
            (RECOVERY, "run_id", SOURCE),
        ]
        for run_id, field, value in cases:
            info = self.client.runs[run_id].info
            old = getattr(info, field)
            setattr(info, field, value)
            with (
                self.subTest(run_id=run_id, field=field),
                self.assertRaises((ValueError, PublicationError)),
            ):
                self.publish()
            setattr(info, field, old)
            self.assertEqual(self.client.writes, [])
        self.client.runs[RECOVERY].data.tags["recovery.source_run_id"] = "wrong"
        with self.assertRaises((ValueError, PublicationError)):
            self.publish()
        self.assertEqual(self.client.writes, [])

    def test_expected_old_hash_is_required_before_any_mutation(self):
        with self.assertRaises((ValueError, PublicationError)):
            publish_replacement(self.client, self.manifest, "0" * 64)
        self.assertEqual(self.client.writes, [])
        self.assertEqual(self.client.tags, [])

    def test_existing_operation_namespace_is_never_overwritten(self):
        self.put(SOURCE, f"{NAMESPACE}/marker", b"existing")
        with self.assertRaises((ValueError, PublicationError)):
            self.publish()
        self.assertEqual(self.client.writes, [])

    def test_missing_oversized_and_malformed_artifacts_fail_before_writes(self):
        for path, replacement in ((CHECKPOINT, None), (MODEL, b"bad: [yaml")):
            previous = self.client.path(RECOVERY, path).read_bytes()
            if replacement is None:
                self.client.path(RECOVERY, path).unlink()
            else:
                self.put(RECOVERY, path, replacement)
            with (
                self.subTest(path=path),
                self.assertRaises((ValueError, PublicationError)),
            ):
                self.publish()
            self.put(RECOVERY, path, previous)
            self.assertEqual(self.client.writes, [])
        with (
            patch("ml.checkpoint_publication.CHECKPOINT_LIMIT", 8),
            self.assertRaises((ValueError, PublicationError)),
        ):
            self.publish()
        self.assertEqual(self.client.writes, [])

    def test_backup_or_stage_readback_failure_blocks_canonical_writes(self):
        for suffix in (
            "backup/checkpoint.ckpt",
            "backup/MLmodel",
            "operation.json",
            "staged/checkpoint.ckpt",
            "staged/MLmodel",
        ):
            with self.subTest(suffix=suffix):
                shutil.rmtree(
                    self.client.path(SOURCE, "checkpoint_repairs"), ignore_errors=True
                )
                self.client.corrupt_read = lambda run, path, suffix=suffix: (
                    path == f"{NAMESPACE}/{suffix}"
                )
                with self.assertRaises(PublicationError) as error:
                    self.publish()
                self.assert_no_canonical_writes()
                self.assertEqual(error.exception.receipt["status"], "failed")
                self.assertTrue(Path(error.exception.journal_path).is_file())

    def test_concurrent_old_change_fails_immediate_recheck(self):
        def change_original(run_id, path):
            if path == f"{NAMESPACE}/staged/MLmodel":
                self.put(SOURCE, CHECKPOINT, b"concurrent writer")

        self.client.before_read = change_original
        with self.assertRaises(PublicationError):
            self.publish()
        self.assert_no_canonical_writes()

    def test_second_canonical_write_failure_reports_partial_and_retains_backup(self):
        self.client.fail_write = lambda run, path: path == MODEL
        with self.assertRaises(PublicationError) as error:
            self.publish()
        self.assertEqual(error.exception.receipt["status"], "partial")
        self.assertEqual(error.exception.receipt["failed_phase"], "replacing_mlmodel")
        self.assertEqual(self.client.path(SOURCE, CHECKPOINT).read_bytes(), self.new)
        self.assertEqual(
            self.client.path(
                SOURCE, f"{NAMESPACE}/backup/checkpoint.ckpt"
            ).read_bytes(),
            self.old,
        )
        self.assertFalse(
            any(
                key == "recovery.publication_status" and value == "completed"
                for _, key, value in self.client.tags
            )
        )

    def test_canonical_readback_failure_does_not_report_success(self):
        count = 0

        def corrupt(run, path):
            nonlocal count
            if run == SOURCE and path == CHECKPOINT:
                count += 1
                return count == 3
            return False

        self.client.corrupt_read = corrupt
        with self.assertRaises(PublicationError) as error:
            self.publish()
        self.assertEqual(error.exception.receipt["status"], "partial")

    def test_empty_operation_namespace_is_never_reused(self):
        self.client.path(SOURCE, NAMESPACE).mkdir(parents=True)
        with self.assertRaises((ValueError, PublicationError)):
            self.publish()
        self.assertEqual(self.client.writes, [])

    def test_invalid_model_metadata_is_rejected_before_writes(self):
        original = self.client.path(RECOVERY, MODEL).read_bytes()
        for change in (
            {"flavors": []},
            {"flavors": {"pytorch": []}},
            {"run_id": SOURCE},
            {"model_size_bytes": 1},
        ):
            model = yaml.safe_load(original)
            model.update(change)
            self.put(RECOVERY, MODEL, yaml.safe_dump(model).encode())
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.publish()
            self.assertEqual(self.client.writes, [])

    def test_checkpoint_callback_mismatch_and_ambiguity_are_rejected(self):
        path = self.root / "altered.ckpt"
        for alteration in ("score", "mode_conflict", "ambiguous", "unsafe", "binding"):
            state = torch.load(
                self.client.path(RECOVERY, CHECKPOINT), weights_only=True
            )
            key = next(iter(state["callbacks"]))
            callback = state["callbacks"][key]
            if alteration == "score":
                callback["best_model_score"] = torch.tensor(0.2)
            elif alteration == "mode_conflict":
                callback["mode"] = "max"
            elif alteration == "ambiguous":
                state["callbacks"][key + " "] = callback.copy()
            elif alteration == "binding":
                state["recovery"] = {"source_run_id": "wrong"}
            else:
                state["unsafe"] = SimpleNamespace(
                    secret="arbitrary pickle is forbidden"
                )
            torch.save(state, path)
            data = path.read_bytes()
            self.put(RECOVERY, CHECKPOINT, data)
            self.manifest.update(checkpoint_sha256=sha(data), checkpoint_size=len(data))
            self.put(RECOVERY, MODEL, self.model_bytes(RECOVERY, len(data)))
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.publish()
            self.assertEqual(self.client.writes, [])
            self.put(RECOVERY, CHECKPOINT, self.new)

    def test_initial_journal_failure_preserves_local_evidence(self):
        self.client.fail_write = lambda run, path: path.endswith("/journal.json")
        with self.assertRaises(PublicationError) as error:
            self.publish()
        self.assert_no_canonical_writes()
        self.assertEqual(error.exception.receipt["status"], "failed")
        self.assertTrue(Path(error.exception.journal_path).is_file())

    def test_persistent_local_journal_failure_preserves_partial_receipt(self):
        original_write = Path.write_bytes
        local_failure = OSError("local journal storage unavailable")

        def fail_after_checkpoint(path, data):
            if (
                path.name == "journal.json"
                and path.parent.name.startswith("checkpoint-publication-journal-")
                and self.client.path(SOURCE, CHECKPOINT).read_bytes() == self.new
            ):
                raise local_failure
            return original_write(path, data)

        with (
            patch.object(Path, "write_bytes", fail_after_checkpoint),
            self.assertRaises(PublicationError) as error,
        ):
            self.publish()
        receipt = error.exception.receipt
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["failed_phase"], "replacing_mlmodel")
        self.assertIs(error.exception.__cause__, local_failure)
        self.assertEqual(receipt["journal_persistence_errors"]["local"], "OSError")
        self.assertTrue(Path(error.exception.journal_path).is_file())
        self.assertEqual(self.client.path(SOURCE, CHECKPOINT).read_bytes(), self.new)
        self.assertEqual(
            self.client.path(
                SOURCE, f"{NAMESPACE}/backup/checkpoint.ckpt"
            ).read_bytes(),
            self.old,
        )
        self.assertTrue(
            self.client.path(SOURCE, f"{NAMESPACE}/backup/MLmodel").is_file()
        )
        self.assertTrue(
            self.client.path(SOURCE, f"{NAMESPACE}/operation.json").is_file()
        )
        remote_journal = json.loads(
            self.client.path(SOURCE, f"{NAMESPACE}/journal.json").read_bytes()
        )
        self.assertEqual(remote_journal["status"], "partial")
        self.assertEqual(
            remote_journal["journal_persistence_errors"]["local"], "OSError"
        )
        self.assertFalse(
            any(
                key == "recovery.publication_status" and value == "completed"
                for _, key, value in self.client.tags
            )
        )

    def test_both_journal_stores_failing_still_raise_publication_error(self):
        original_write = Path.write_bytes
        local_failure = OSError("local journal storage unavailable")

        def fail_journals(path, data):
            if (
                path.name == "journal.json"
                and self.client.path(SOURCE, CHECKPOINT).read_bytes() == self.new
            ):
                raise local_failure
            return original_write(path, data)

        with (
            patch.object(Path, "write_bytes", fail_journals),
            self.assertRaises(PublicationError) as error,
        ):
            self.publish()
        receipt = error.exception.receipt
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["failed_phase"], "replacing_mlmodel")
        self.assertIs(error.exception.__cause__, local_failure)
        self.assertEqual(
            receipt["journal_persistence_errors"],
            {"local": "OSError", "remote": "OSError"},
        )
        self.assertEqual(receipt["namespace"], NAMESPACE)
        self.assertTrue(
            self.client.path(SOURCE, f"{NAMESPACE}/backup/checkpoint.ckpt").is_file()
        )
        self.assertEqual(
            self.client.runs[SOURCE].data.tags["recovery.publication_status"], "partial"
        )

    def test_final_journal_failure_cannot_leave_completed_marker(self):
        original_upload = self.client.log_artifact

        def upload(run, local, artifact_path):
            if (
                Path(local).name == "journal.json"
                and json.loads(Path(local).read_text())["status"] == "completed"
            ):
                raise OSError("final journal unavailable")
            original_upload(run, local, artifact_path)

        self.client.log_artifact = upload
        with self.assertRaises(PublicationError):
            self.publish()
        self.assertFalse(
            any(
                key == "recovery.publication_status" and value == "completed"
                for _, key, value in self.client.tags
            )
        )

    def test_source_manifest_or_status_change_during_staging_blocks_overwrite(self):
        old_model = self.client.path(SOURCE, MODEL).read_bytes()
        for alteration in ("model", "status"):
            with self.subTest(alteration=alteration):
                shutil.rmtree(
                    self.client.path(SOURCE, "checkpoint_repairs"), ignore_errors=True
                )
                self.client.writes.clear()

                def change(run_id, path, alteration=alteration):
                    if path != f"{NAMESPACE}/staged/MLmodel":
                        return
                    if alteration == "model":
                        self.put(SOURCE, MODEL, b"concurrent model change")
                    else:
                        self.client.runs[SOURCE].info.status = "RUNNING"

                self.client.before_read = change
                with self.assertRaises(PublicationError):
                    self.publish()
                self.assert_no_canonical_writes()
                self.put(SOURCE, MODEL, old_model)
                self.client.runs[SOURCE].info.status = "FINISHED"

    def test_tracking_endpoint_must_be_available_and_credentials_free(self):
        original = self.client.tracking_uri
        for uri in (
            None,
            "https://user:secret@mlflow.example.test",
            "https://mlflow.example.test?secret=x",
        ):
            self.client.tracking_uri = uri
            if uri is not None:
                self.manifest["tracking_uri"] = uri
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                self.publish()
            self.assertEqual(self.client.writes, [])
        self.client.tracking_uri = original

    def test_max_objective_checks_direction(self):
        self.manifest.update(mode="max", target_score=0.4)
        with self.assertRaises((ValueError, PublicationError)):
            self.publish()
        self.assertEqual(self.client.writes, [])


if __name__ == "__main__":
    unittest.main()
