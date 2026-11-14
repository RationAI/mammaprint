import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from ml.concept_explainability.model import (
    SourceRun,
    load_frozen_model,
    load_source_run,
    model_outputs,
    verify_training_revision,
)
from ml.models.aggregators.mean import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads.linear import LinearHead
from ml.models.module import MammaprintModule


BASE = "e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2"


# Original direct-download implementation, before resumable transport helpers.
HISTORICAL_TRANSPORT = '''import time

from mlflow.artifacts import download_artifacts as _download_artifacts

from mlflow.exceptions import MlflowException

def download_artifacts_with_retries(
    artifact_uri: str,
    *,
    max_retries: int = MLFLOW_DOWNLOAD_MAX_RETRIES,
    initial_backoff_seconds: float = MLFLOW_DOWNLOAD_INITIAL_BACKOFF_SECONDS,
) -> Path:
    """Download an MLflow artifact, retrying transient download failures.

    MLflow raises one aggregate :class:`MlflowException` when any file in a
    directory fails to download. Retrying the directory lets a short-lived
    storage or network failure recover instead of aborting dataset setup.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if initial_backoff_seconds < 0:
        raise ValueError("initial_backoff_seconds must be non-negative.")

    for retry in range(max_retries + 1):
        try:
            return Path(_download_artifacts(artifact_uri=artifact_uri))
        except MlflowException:
            if retry == max_retries:
                raise

            delay = initial_backoff_seconds * 2**retry
            logger.warning(
                "MLflow artifact download failed for %s; retrying in %.1f seconds "
                "(%d/%d).",
                artifact_uri,
                delay,
                retry + 1,
                max_retries,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")'''


def tiny_config(out_dim=1):
    return OmegaConf.create(
        {
            "label_mode": "both" if out_dim == 2 else "type",
            "ml": {
                "_target_": "ml.models.MammaprintModule",
                "encoder": {
                    "_target_": "ml.models.encoders.IdentityEncoder",
                    "out_dim": 4,
                },
                "aggregator": {
                    "_target_": "ml.models.aggregators.MeanPool",
                    "feature_dim": 4,
                },
                "head": {
                    "_target_": "ml.models.heads.LinearHead",
                    "in_dim": 4,
                    "out_dim": out_dim,
                },
            },
        }
    )


def embedding_config():
    config = tiny_config()
    config.datamodule = {
        "_target_": "ml.data.DataModule",
        **{
            split: {
                "_target_": "ml.data.SingleScaleDataset",
                "split": split,
                "label_mode": "type",
                "data_mapping": "/unused/labels.csv",
                "levels": {3: {"uris": {split: f"/unused/{split}"}}},
            }
            for split in ("train", "val", "test")
        },
    }
    return config


class RevisionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        repository = Path(__file__).resolve().parents[1]
        for relative in ("ml/models", "ml/data"):
            shutil.copytree(repository / relative, self.root / relative)
        shutil.copyfile(repository / "ml/typing.py", self.root / "ml/typing.py")
        self.sources = self.root / "ml/data/datasets/_sources.py"
        current = self.sources.read_text()
        tree = ast.parse(current)
        transport_functions = {
            "download_artifacts_with_retries",
            "_artifact_files",
            "_is_complete_download",
            "_download_incomplete_artifacts",
        }
        transport_modules = {
            "time",
            "tempfile",
            "mlflow.artifacts",
            "mlflow.entities",
            "mlflow.exceptions",
            "mlflow.tracking.artifact_utils",
            "mlflow.utils.uri",
        }
        core = [
            node
            for node in tree.body
            if not (
                (isinstance(node, ast.FunctionDef) and node.name in transport_functions)
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module in transport_modules
                )
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name in transport_modules for alias in node.names)
                )
            )
        ]
        # Keep all scientific functions/imports identical while restoring the real
        # historical transfer implementation. This works without fetched objects.
        self.sources.write_text(
            ast.unparse(ast.Module(body=core, type_ignores=[]))
            + "\n"
            + HISTORICAL_TRANSPORT
            + "\n"
        )
        self.git("init", "-q")
        self.git("add", "ml")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Historical fixture",
        )
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.sources.write_text(current)
        self.patch = patch("ml.concept_explainability.model.REPOSITORY", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout

    def test_reviewed_transport_drift_accepts_precomputed_config(self):
        verify_training_revision(self.revision, config=embedding_config())

    def test_transport_drift_without_config_remains_rejected(self):
        with self.assertRaises(RuntimeError):
            verify_training_revision(self.revision)

    def test_other_data_source_drift_requires_exact_match(self):
        raw_tiles = self.root / "ml/data/datasets/raw_tiles.py"
        raw_tiles.write_text(
            raw_tiles.read_text() + "\n# Unused raw-tile maintenance.\n"
        )
        with self.assertRaises(RuntimeError):
            verify_training_revision(self.revision, config=embedding_config())

    def test_eager_data_dependency_cannot_override_selected_dataset(self):
        spatial = self.root / "ml/data/datasets/spatial_scale.py"
        spatial.write_text(
            spatial.read_text()
            + "\nSingleScaleDataset.__getitem__ = "
            + 'lambda self, idx: ("changed-input", 0, {})\n'
        )
        # ml.data eagerly imports spatial_scale even for SingleScaleDataset.
        # Its module body can therefore change the selected class at import.
        with self.assertRaises(RuntimeError):
            verify_training_revision(self.revision, config=embedding_config())

    def test_unknown_revision_remains_rejected(self):
        with self.assertRaises(RuntimeError):
            verify_training_revision("0" * 40, config=embedding_config())

    def test_changed_scientific_behavior_is_rejected(self):
        changes = {
            "ml/models/aggregators/mean.py": ("bag.mean(dim=0)", "bag.sum(dim=0)"),
            "ml/data/datasets/labels.py": ("return 1", "return 0"),
            "ml/data/datasets/single_scale.py": (
                'slides["name"].isin(present)',
                '~slides["name"].isin(present)',
            ),
            "ml/data/datasets/_sources.py": ("uris.get(split)", 'uris.get("train")'),
        }
        for relative, (before, after) in changes.items():
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                self.assertIn(before, original)
                path.write_text(original.replace(before, after))
                try:
                    with self.assertRaises(RuntimeError):
                        verify_training_revision(
                            self.revision, config=embedding_config()
                        )
                finally:
                    path.write_text(original)

    def test_unreviewed_download_behavior_is_rejected(self):
        self.sources.write_text(
            self.sources.read_text().replace(
                "artifact_uri=artifact_uri,", 'artifact_uri="/different-corpus",'
            )
        )
        with self.assertRaises(RuntimeError):
            verify_training_revision(self.revision, config=embedding_config())

    def test_unsupported_configured_paths_are_rejected(self):
        for path, target in (
            ("ml.encoder", "ml.models.encoders.VGG16Encoder"),
            ("ml.aggregator", "custom.MeanPool"),
            ("ml.head", "custom.LinearHead"),
            ("datamodule.train", "ml.data.RawTileSlideDataset"),
            ("datamodule.val", "ml.data.SpatialScaleDataset"),
        ):
            with self.subTest(path=path):
                config = embedding_config()
                OmegaConf.update(config, path + "._target_", target)
                with self.assertRaises(RuntimeError):
                    verify_training_revision(self.revision, config=config)

    def test_nonidentity_activation_is_rejected_before_checkpoint_download(self):
        config = embedding_config()
        config.ml.output_activation = {"_target_": "torch.nn.Softmax", "dim": -1}
        with self.assertRaises(RuntimeError):
            verify_training_revision(self.revision, config=config)

    def test_local_source_cli_checks_configured_compatibility(self):
        from ml.concepts import run

        source_config = self.root / "resolved.yaml"
        OmegaConf.save(embedding_config(), source_config)
        config = OmegaConf.create(
            {
                "action": "fit",
                "split": "train",
                "log_to_mlflow": False,
                "output_dir": str(self.root / "output"),
                "local_source_config": str(source_config),
                "local_checkpoint": None,
                "source_run_id": "local-fixture",
                "training_sha": self.revision,
            }
        )
        # Stop at the data boundary: no embedding/WSI or external work is needed
        # to prove the real CLI reached its audit with compatible provenance.
        with (
            patch("ml.concepts.audit_splits", side_effect=LookupError("audit reached")),
            self.assertRaisesRegex(LookupError, "audit reached"),
        ):
            run(config)


class SourceTests(unittest.TestCase):
    def test_copied_numerical_bodies_match_pinned_source(self):
        root = Path(__file__).resolve().parents[1]
        provenance = json.loads(
            (root / "third_party/ibd-explainability-symbols.json").read_text()
        )
        tree = ast.parse((root / "ml/concept_explainability/upstream.py").read_text())
        nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for name, metadata in provenance.items():
            digest = hashlib.sha256(
                ast.dump(nodes[name], include_attributes=False).encode()
            ).hexdigest()
            self.assertEqual(digest, metadata["ast_sha256"], name)

    def test_source_run_load_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "resolved.yaml"
            OmegaConf.save(tiny_config(), config)
            checkpoint = root / "checkpoint.ckpt"
            checkpoint.write_bytes(b"test checkpoint")
            client = Mock()
            client.get_run.return_value = SimpleNamespace(
                data=SimpleNamespace(tags={"mlflow.source.git.commit": BASE})
            )
            client.download_artifacts.side_effect = [str(config), str(checkpoint)]
            result = load_source_run("a" * 32, root / "cache", client=client)
            self.assertEqual(result.training_sha, BASE)
            self.assertEqual(
                result.checkpoint_sha256, hashlib.sha256(b"test checkpoint").hexdigest()
            )
            self.assertEqual(
                [c[0] for c in client.method_calls],
                ["get_run", "download_artifacts", "download_artifacts"],
            )

    def test_missing_resolved_config_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            client.get_run.return_value = SimpleNamespace(
                data=SimpleNamespace(tags={"mlflow.source.git.commit": BASE})
            )
            client.download_artifacts.side_effect = FileNotFoundError("missing")
            with self.assertRaisesRegex(RuntimeError, "resolved"):
                load_source_run("a" * 32, Path(directory), client=client)

    def test_unresolved_training_revision_fails(self):
        with self.assertRaisesRegex(ValueError, "training.*revision"):
            verify_training_revision(None)

    def test_checkpoint_requires_exact_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.ckpt"
            torch.save({"state_dict": {"wrong.weight": torch.zeros(1)}}, path)
            source = SourceRun("test", BASE, tiny_config(), path, "checksum")
            with self.assertRaises(RuntimeError):
                load_frozen_model(source)

    def test_parameter_free_output_activation_is_rejected(self):
        from hydra.utils import instantiate

        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config()
            config.ml.output_activation = {"_target_": "torch.nn.Softmax", "dim": -1}
            original = instantiate(config.ml)
            path = Path(directory) / "checkpoint.ckpt"
            torch.save({"state_dict": original.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Identity.*output_activation"):
                load_frozen_model(SourceRun("test", BASE, config, path, "checksum"))

    def test_joint_outputs_are_logit_and_index(self):
        module = MammaprintModule(
            IdentityEncoder(out_dim=4), MeanPool(4), LinearHead(4, out_dim=2, dropout=0)
        )
        with torch.no_grad():
            module.head.linear.weight.zero_()
            module.head.linear.bias.copy_(torch.tensor([2.0, -0.7]))
        result = model_outputs(module, torch.ones(3, 4), "both")
        torch.testing.assert_close(result, torch.tensor([2.0, -0.7]))

    def test_frozen_checkpoint_matches_native_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            from hydra.utils import instantiate

            config = tiny_config(2)
            original = instantiate(config.ml).eval()
            path = Path(directory) / "checkpoint.ckpt"
            torch.save({"state_dict": original.state_dict()}, path)
            loaded = load_frozen_model(
                SourceRun("test", BASE, config, path, "checksum")
            )
            bag = torch.randn(7, 4)
            torch.testing.assert_close(
                model_outputs(loaded, bag, "both"),
                original(bag)[0],
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
