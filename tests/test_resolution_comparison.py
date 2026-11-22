import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.ml import evaluate_resolution_comparison as evaluate
from scripts.ml import run_resolution_comparison as run


class ResolutionComparisonTest(unittest.TestCase):
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

    def test_metrics_use_raw_tied_scores_and_classify_zero_as_positive(self):
        targets = np.array([-1.0, 0.0, 1.0, 1.0])
        scores = np.array([0.1, 0.2, 0.2, -0.5])
        mae, spearman, auc = evaluate.metrics(targets, scores)
        self.assertAlmostEqual(mae, 0.9)
        self.assertAlmostEqual(spearman, spearmanr(targets, scores).statistic)
        self.assertAlmostEqual(auc, 2 / 3)
        self.assertTrue(np.isnan(evaluate.metrics(np.array([0.0, 1.0]), scores[:2])[2]))

    def test_paired_bootstrap_resamples_shared_patients_and_preserves_zero_difference(
        self,
    ):
        target = np.arange(-3.0, 3.0)
        vectors = np.stack([target, target + 0.1, target + 0.2])
        groups = ["p0", "p0", "p1", "p1", "p2", "p2"]
        scores = np.stack([vectors, vectors])
        intervals = evaluate.paired_intervals(target, scores, groups, 100)
        self.assertEqual(intervals["mae"]["defined_samples"], 100)
        for metric in evaluate.METRICS:
            self.assertEqual(intervals[metric]["ci95"], [0.0, 0.0])

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

    def test_validation_never_reads_test_and_only_locked_test_runs_once(self):
        with tempfile.TemporaryDirectory() as workdir:
            self._exercise_scoring_workflow(Path(workdir))

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

    def _exercise_scoring_workflow(self, output: Path):
        mapping = output / "data_mapping.csv"
        targets = [-0.5, 0.0, 0.3, 0.5]
        splits = ("val", "test")
        manifest = {
            "digest": "bundle",
            "splits": {
                split: {f"{split}{index}": 1 for index in range(len(targets))}
                for split in splits
            },
            "files": {
                "l1": {"test": {f"test{i}": "hash" for i in range(len(targets))}}
            },
        }
        mapping.write_text(
            "path,split,mammaprint_index\n"
            + "".join(
                f"/{split}{index}.svs,{split},{target}\n"
                for split in splits
                for index, target in enumerate(targets)
            )
        )
        args = argparse.Namespace(
            bundle=output,
            data_mapping=mapping,
            output=output,
            patient_column=None,
            bootstrap_samples=30,
            gpu_hours=20.0,
            budget_gpu_hours=40.0,
        )
        for fit in run.FITS:
            (output / fit.name).mkdir()
        observed = []

        def fake_record(_output, fit, _digest):
            return ({"fit": fit._asdict(), "mlflow_run_id": fit.name}, output, output)

        def fake_prediction(fit, split, _directory, rows, _checkpoint, _config):
            observed.append((fit, split))
            return pd.DataFrame(
                {
                    "slide_id": sorted(rows),
                    "target": targets,
                    "prediction": np.asarray(targets)
                    + (0.01 * fit.seed if fit.level == 1 else 0.25),
                }
            )

        with (
            patch.object(evaluate, "_fit_record", side_effect=fake_record),
            patch.object(evaluate, "_predict_frame", side_effect=fake_prediction),
            patch.object(evaluate, "split_rows", wraps=run.split_rows) as split_rows,
            patch.object(evaluate, "verify_split") as verify_split,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            evaluate._validation(args, manifest)
            self.assertEqual(observed, [(fit, "val") for fit in run.FITS])
            self.assertEqual([call.args[-1] for call in split_rows.mock_calls], ["val"])
            self.assertEqual(
                [call.args[2] for call in verify_split.mock_calls], ["val", "val"]
            )
            decision = json.loads((output / "decision.json").read_text())
            self.assertEqual(decision["decision"], "go")
            self.assertEqual(
                decision["fit"], {"level": 1, "pooling": "mean", "seed": 0}
            )

            verify_split.side_effect = ValueError("Invalid test bag")
            with self.assertRaisesRegex(ValueError, "Invalid test bag"):
                evaluate._test(args, manifest)
            self.assertFalse((output / "test_started.json").exists())
            verify_split.side_effect = None
            evaluate._test(args, manifest)
            self.assertEqual(observed[-1], (run.Fit(1, "mean", 0), "test"))
            self.assertEqual(verify_split.call_args.args[1:3], (1, "test"))
            with self.assertRaisesRegex(RuntimeError, "already been attempted"):
                evaluate._test(args, manifest)
            self.assertEqual(len(observed), 13)
