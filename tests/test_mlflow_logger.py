"""Local artifact-client and native CPU regressions for checkpoint uploads."""

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from rationai.mlkit.lightning.loggers import MLFlowLogger as ParentLogger
from torch.utils.data import DataLoader, TensorDataset


class ArtifactClient:
    """Replace only the MLflow transport, preserving actual uploaded file bytes."""

    def __init__(self, root):
        self.root = Path(root)
        self.uploads = []
        self.fail_upload = False
        self.fail_delete = False
        self._tracking_client = self

    def run_root(self, run_id):
        return self.root / run_id

    def list_artifacts(self, run_id, path):
        root = self.run_root(run_id) / path
        if not root.exists():
            return []
        return [
            SimpleNamespace(path=f"{path}/{p.name}", is_dir=p.is_dir())
            for p in root.iterdir()
        ]

    def log_artifacts(self, run_id, local_dir, artifact_path):
        target = self.run_root(run_id) / artifact_path
        shutil.copytree(local_dir, target, dirs_exist_ok=True)
        if self.fail_upload:
            self.fail_upload = False
            raise OSError("partial upload")
        self.uploads.append(
            (run_id, artifact_path, (target / "checkpoint.ckpt").read_bytes())
        )

    def _get_artifact_repo(self, run_id):
        return SimpleNamespace(
            delete_artifacts=lambda path: self.delete_artifacts(run_id, path)
        )

    def delete_artifacts(self, run_id, path):
        if self.fail_delete:
            self.fail_delete = False
            raise OSError("delete failed")
        shutil.rmtree(self.run_root(run_id) / path)

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_batch(self, **kwargs):
        pass

    def set_terminated(self, run_id, status):
        pass


def make_logger(client, logger_class=ParentLogger, log_model="all", run_id="run"):
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        logger = logger_class(
            tracking_uri=client.root.as_uri(),
            experiment_name="local",
            log_model=log_model,
        )
    logger._initialized = True
    logger._run_id = run_id
    logger._experiment_id = "experiment"
    return logger


def callback_for(path, *, last="", save_top_k=1):
    return SimpleNamespace(
        best_model_path=str(path),
        best_k_models={str(path): torch.tensor(0.1)},
        last_model_path=str(last),
        save_top_k=save_top_k,
        _last_checkpoint_saved=str(path),
    )


class ParentRegressionTests(unittest.TestCase):
    def test_parent_skips_rewritten_best(self):
        """Characterize the bug: a remote best stem masks changed local bytes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ArtifactClient(root / "artifacts")
            logger = make_logger(client)
            best = root / "best.ckpt"
            best.write_bytes(b"epoch zero")
            callback = callback_for(best)
            logger.after_save_checkpoint(callback)
            best.write_bytes(b"epoch two!")
            logger.after_save_checkpoint(callback)
            remote = client.run_root("run") / "checkpoints/best/checkpoint.ckpt"
            self.assertEqual(remote.read_bytes(), b"epoch zero")
            self.assertNotEqual(remote.read_bytes(), best.read_bytes())


class LoggerTests(unittest.TestCase):
    def setUp(self):
        from ml.loggers import MLFlowLogger

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.client = ArtifactClient(self.root / "artifacts")
        self.logger_class = MLFlowLogger
        self.logger = make_logger(self.client, MLFlowLogger)
        self.best = self.root / "best.ckpt"
        self.best.write_bytes(b"epoch zero")
        self.callback = callback_for(self.best)

    def remote(self, stem="best", run="run"):
        return self.client.run_root(run) / f"checkpoints/{stem}/checkpoint.ckpt"

    def test_same_path_rewrite_replaces_payload_and_manifest(self):
        self.logger.after_save_checkpoint(self.callback)
        self.best.write_bytes(b"epoch two payload")
        self.logger.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote().read_bytes(), b"epoch two payload")
        manifest = OmegaConf.load(self.remote().with_name("MLmodel"))
        self.assertEqual(manifest.model_size_bytes, 17)
        self.assertEqual(manifest.flavors.pytorch.model_data, "checkpoint.ckpt")
        self.assertEqual(len(self.client.uploads), 2)

    def test_unchanged_and_touched_bytes_skip_upload(self):
        self.logger.after_save_checkpoint(self.callback)
        with patch(
            "ml.loggers._checkpoint_sha256", side_effect=AssertionError("rehash")
        ):
            self.logger._scan_and_log_checkpoints(self.callback)
        self.best.touch()
        self.logger._scan_and_log_checkpoints(self.callback)
        self.assertEqual(len(self.client.uploads), 1)

    def test_saved_path_forces_hash_even_when_stat_signature_matches(self):
        self.logger.after_save_checkpoint(self.callback)
        signature = self.best.stat()
        self.best.write_bytes(b"epoch two!")
        os.utime(self.best, ns=(signature.st_atime_ns, signature.st_mtime_ns))
        with patch("ml.loggers._checkpoint_stat", return_value=(1, 2, 10, 4)):
            self.logger.after_save_checkpoint(self.callback)
            self.best.write_bytes(b"epoch six!")
            self.logger.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote().read_bytes(), b"epoch six!")
        self.assertEqual(len(self.client.uploads), 3)

    def test_retained_checkpoints_do_not_rehash_on_unrelated_save(self):
        self.logger.after_save_checkpoint(self.callback)
        other = self.root / "other.ckpt"
        other.write_bytes(b"other")
        self.callback.best_k_models[str(other)] = torch.tensor(0.2)
        self.callback._last_checkpoint_saved = str(other)
        from ml.loggers import _checkpoint_sha256

        with patch("ml.loggers._checkpoint_sha256", wraps=_checkpoint_sha256) as digest:
            self.logger.after_save_checkpoint(self.callback)
        self.assertEqual([call.args[0] for call in digest.call_args_list], [other])

    def test_fresh_logger_and_new_run_do_not_trust_remote_stem(self):
        self.logger.after_save_checkpoint(self.callback)
        self.remote().write_bytes(b"stale")
        fresh = make_logger(self.client, self.logger_class)
        fresh.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote().read_bytes(), b"epoch zero")
        self.logger._run_id = "different"
        self.logger.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote(run="different").read_bytes(), b"epoch zero")

    def test_deleted_remote_stem_is_recreated(self):
        self.logger.after_save_checkpoint(self.callback)
        shutil.rmtree(self.remote().parent)
        self.logger._scan_and_log_checkpoints(self.callback)
        self.assertEqual(self.remote().read_bytes(), b"epoch zero")
        self.assertEqual(len(self.client.uploads), 2)

    def test_partial_upload_failure_retries_and_is_not_finalized(self):
        self.logger.after_save_checkpoint(self.callback)
        self.best.write_bytes(b"epoch two!")
        self.client.fail_upload = True
        with self.assertRaisesRegex(OSError, "partial upload"):
            self.logger.after_save_checkpoint(self.callback)
        self.logger.finalize()
        self.assertEqual(self.remote().read_bytes(), b"epoch two!")
        self.assertEqual(len(self.client.uploads), 2)

    def test_same_stat_failed_replacement_is_retried_at_finalize(self):
        with patch("ml.loggers._checkpoint_stat", return_value=(1, 2, 10, 4)):
            self.logger.after_save_checkpoint(self.callback)
            self.best.write_bytes(b"epoch two!")
            with (
                patch.object(
                    self.client, "log_artifacts", side_effect=OSError("upload failed")
                ),
                self.assertRaisesRegex(OSError, "upload failed"),
            ):
                self.logger.after_save_checkpoint(self.callback)
            self.assertEqual(self.remote().read_bytes(), b"epoch zero")
            self.logger.finalize()
        self.assertEqual(self.remote().read_bytes(), b"epoch two!")
        self.assertEqual(len(self.client.uploads), 2)

    def test_pruning_failure_is_visible_and_retried(self):
        self.logger.after_save_checkpoint(self.callback)
        other = self.root / "other.ckpt"
        other.write_bytes(b"other")
        callback = callback_for(other)
        self.client.fail_delete = True
        with self.assertRaisesRegex(OSError, "delete failed"):
            self.logger.after_save_checkpoint(callback)
        self.assertTrue(self.remote().exists())
        self.logger.finalize()
        self.assertFalse(self.remote().exists())
        self.assertEqual(self.remote("other").read_bytes(), b"other")
        self.logger.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote().read_bytes(), b"epoch zero")

    def test_last_changes_upload_but_unchanged_last_skips(self):
        last = self.root / "last.ckpt"
        last.write_bytes(b"last zero")
        self.callback.last_model_path = str(last)
        self.logger.after_save_checkpoint(self.callback)
        self.logger._scan_and_log_checkpoints(self.callback)
        self.assertEqual(len(self.client.uploads), 2)
        last.write_bytes(b"last next")
        self.logger.finalize()
        self.assertEqual(self.remote("last").read_bytes(), b"last next")
        self.assertEqual(len(self.client.uploads), 3)

    def test_missing_last_symlink_is_deferred_until_final_flush(self):
        last = self.root / "last.ckpt"
        target = self.root / "next.ckpt"
        last.symlink_to(target)
        self.callback.last_model_path = str(last)
        self.logger.after_save_checkpoint(self.callback)
        self.assertFalse(self.remote("last").exists())
        target.write_bytes(b"linked final")
        self.logger.finalize()
        self.assertEqual(self.remote("last").read_bytes(), b"linked final")

    def test_missing_explicitly_saved_checkpoint_fails(self):
        self.best.unlink()
        with self.assertRaises(FileNotFoundError):
            self.logger.after_save_checkpoint(self.callback)

    def test_disabled_and_nonzero_rank_make_no_artifact_calls(self):
        disabled = make_logger(self.client, self.logger_class, log_model=False)
        with patch.object(
            self.client, "list_artifacts", side_effect=AssertionError("scan")
        ):
            disabled.after_save_checkpoint(self.callback)
            disabled.finalize()
            with patch.object(rank_zero_only, "rank", 1):
                self.logger.after_save_checkpoint(self.callback)
                self.logger._scan_and_log_checkpoints(self.callback)
                self.logger.finalize()
        self.assertFalse(self.remote().exists())

    def test_true_defers_top_one_and_immediately_uploads_save_all(self):
        deferred = make_logger(self.client, self.logger_class, log_model=True)
        deferred.after_save_checkpoint(self.callback)
        self.assertFalse(self.remote().exists())
        deferred.finalize()
        self.assertEqual(self.remote().read_bytes(), b"epoch zero")
        immediate = make_logger(
            self.client, self.logger_class, log_model=True, run_id="all"
        )
        self.callback.save_top_k = -1
        immediate.after_save_checkpoint(self.callback)
        self.assertEqual(self.remote(run="all").read_bytes(), b"epoch zero")

    def test_configuration_instantiates_repaired_logger(self):
        import hydra

        config = OmegaConf.load(
            Path(__file__).parents[1] / "configs/logger/mlflow.yaml"
        )
        config.experiment_name = "local"
        config.run_name = "local"
        config.tags = {}
        config.tracking_uri = self.client.root.as_uri()
        with patch("mlflow.tracking.MlflowClient", return_value=self.client):
            logger = hydra.utils.instantiate(config)
        self.assertIsInstance(logger, self.logger_class)


class TinyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("valid/loss", [0.3, 0.2, 0.1, 0.4][self.current_epoch])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class NativeTrainerTests(unittest.TestCase):
    def fit(self, root, save_last):
        from ml.loggers import MLFlowLogger

        client = ArtifactClient(root / "artifacts")
        logger = make_logger(client, MLFlowLogger)
        checkpoint = ModelCheckpoint(
            dirpath=root / "checkpoints",
            filename="best",
            monitor="valid/loss",
            mode="min",
            save_top_k=1,
            save_last=save_last,
            enable_version_counter=False,
            # Pinned Lightning saves default last only after a top-k save.
            # This step cadence defers to validation, writing last every epoch.
            every_n_train_steps=1,
            save_on_train_epoch_end=False,
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=4,
            logger=logger,
            callbacks=[checkpoint],
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        data = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
        trainer.fit(TinyModule(), train_dataloaders=data, val_dataloaders=data)
        return client, checkpoint

    def test_native_trainer_uploads_final_best_and_distinct_last_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, callback = self.fit(Path(tmp), True)
            loaded = {}
            for stem, path, epoch in (
                ("best", callback.best_model_path, 2),
                ("last", callback.last_model_path, 3),
            ):
                local = Path(path).read_bytes()
                remote = (
                    client.run_root("run") / f"checkpoints/{stem}/checkpoint.ckpt"
                ).read_bytes()
                self.assertEqual(
                    hashlib.sha256(remote).digest(), hashlib.sha256(local).digest()
                )
                saved = torch.load(path, weights_only=False, map_location="cpu")
                self.assertEqual(saved["epoch"], epoch)
                self.assertEqual(saved["global_step"], epoch + 1)
                loaded[stem] = saved
                remote_state = torch.load(
                    client.run_root("run") / f"checkpoints/{stem}/checkpoint.ckpt",
                    weights_only=False,
                    map_location="cpu",
                )
                self.assertTrue(
                    torch.equal(
                        saved["state_dict"]["weight"],
                        remote_state["state_dict"]["weight"],
                    )
                )
                score = next(iter(saved["callbacks"].values()))["best_model_score"]
                self.assertAlmostEqual(score.item(), 0.1)
            self.assertNotEqual(
                loaded["best"]["state_dict"]["weight"].item(),
                loaded["last"]["state_dict"]["weight"].item(),
            )
            self.assertAlmostEqual(loaded["best"]["state_dict"]["weight"].item(), 0.512)
            self.assertAlmostEqual(
                loaded["last"]["state_dict"]["weight"].item(), 0.4096
            )
            snapshots = {
                payload
                for _, path, payload in client.uploads
                if path == "checkpoints/best"
            }
            self.assertGreaterEqual(len(snapshots), 3)

    def test_native_last_link_is_flushed_after_final_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, callback = self.fit(Path(tmp), "link")
            last = Path(callback.last_model_path)
            self.assertTrue(last.is_symlink())
            remote = client.run_root("run") / "checkpoints/last/checkpoint.ckpt"
            self.assertEqual(remote.read_bytes(), last.read_bytes())
            self.assertEqual(torch.load(remote, weights_only=False)["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
