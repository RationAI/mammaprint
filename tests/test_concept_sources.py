import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
