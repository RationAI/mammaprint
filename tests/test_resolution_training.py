import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from scripts.ml import run_resolution_training as run


class ResolutionTrainingTest(unittest.TestCase):
    def test_matrix_has_no_test_loader_or_post_fit_test(self):
        self.assertEqual(len(run.FITS), 12)
        self.assertEqual(len(set(run.FITS)), 12)
        for fit in (run.Fit(1, "mean", 0), run.Fit(2, "attention", 2)):
            config = run._compose_fit(
                run.fit_overrides(
                    fit, Path("/tmp/bundle"), Path("/tmp/attempt"), "test"
                )
            )
            self.assertFalse(config.test_after_fit)
            self.assertNotIn("test", config.datamodule)
            self.assertEqual(config.seed, fit.seed)
            self.assertEqual(config.ml.loss._target_, "torch.nn.MSELoss")
            self.assertEqual(config.ml.head._target_, "ml.models.heads.LinearHead")
            self.assertEqual(set(config.dataset.levels.keys()), {fit.level})

    def test_validation_metrics_rank_continuous_scores_and_use_index_zero(self):
        import torch
        from hydra.utils import instantiate

        config = run._compose_fit(
            run.fit_overrides(
                run.Fit(1, "mean", 0), Path("/tmp/bundle"), Path("/tmp/attempt"), "test"
            )
        )
        model = instantiate(config.ml)
        targets = [-1.0, 0.0, 1.0, 1.0]
        scores = [0.1, 0.2, 0.2, -0.5]
        for score, target in zip(scores, targets, strict=True):
            model.valid_metrics.update(
                torch.tensor([[score]]), torch.tensor([[target]])
            )
        metrics = model.valid_metrics.compute()
        self.assertAlmostEqual(metrics["valid/mae"].item(), 0.9)
        self.assertAlmostEqual(
            metrics["valid/spearman"].item(), spearmanr(targets, scores).statistic
        )
        self.assertAlmostEqual(metrics["valid/sign_auroc"].item(), 2 / 3)

    def test_dataset_reuses_local_bag_for_dotted_slide_stem(self):
        from ml.data import SingleScaleDataset

        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            mapping = root / "mapping.csv"
            mapping.write_text("path,split,mammaprint_index\n/slide.scan.svs,val,0.2\n")
            embeddings = root / "embeddings"
            embeddings.mkdir()
            pd.DataFrame({"embedding": [[0.0, 1.0]]}).to_parquet(
                embeddings / "slide.scan.parquet"
            )
            dataset = SingleScaleDataset(
                "val",
                {1: {"uris": {"val": embeddings.as_uri()}}},
                mapping,
                "index",
            )
            self.assertEqual(dataset.embeddings_dir, embeddings)
            embedding, label, metadata = dataset[0]
            self.assertEqual(embedding.tolist(), [[0.0, 1.0]])
            self.assertAlmostEqual(label.item(), 0.2)
            self.assertEqual(metadata["slide_id"], "slide.scan")

    def test_manifest_digest_detects_changed_cohort(self):
        with tempfile.TemporaryDirectory() as workdir:
            bundle = Path(workdir)
            mapping = bundle / "data_mapping.csv"
            mapping.write_text("path,split,mammaprint_index\n/slide.svs,train,0.2\n")
            manifest = {
                "data_mapping_digest": run.file_digest(mapping),
                "splits": {"train": {"slide": 2}},
            }
            manifest["digest"] = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(run.load_manifest(bundle, mapping), manifest)
            manifest["splits"]["train"]["slide"] = 1
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "invalid digest"):
                run.load_manifest(bundle, mapping)
