"""Safety/identity tests for the bounded real-asset local pilot (no network)."""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.ml import run_local_concept_pilot as pilot


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}


class PilotSafetyTests(unittest.TestCase):
    def test_dry_run_never_downloads_computes_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "assets.json"
            assets = [
                {
                    "slide_id": f"slide{i}",
                    "split": split,
                    "source_uri": f"mlflow-artifacts:/3/source/artifacts/{split}/slide{i}.parquet",
                    "local_path": str(root / f"slide{i}.parquet"),
                    "sha256": "a" * 64,
                    "bytes": 10,
                    "rows": 16,
                }
                for i, split in enumerate(["train", "train", "val", "val"])
            ]
            manifest.write_text(json.dumps(assets))
            argv = [
                "--tracking-uri",
                "https://mlflow.example",
                "--wsi-base-url",
                "https://wsi.example/v3",
                "--source-run-id",
                "run",
                "--expected-experiment-id",
                "3",
                "--expected-experiment-name",
                "MammaPrint",
                "--assets-manifest",
                str(manifest),
                "--output-root",
                str(root / "output"),
                "--cache-root",
                str(root / "cache"),
            ]
            for name in ("source-config", "checkpoint", "slides"):
                argv += [f"--{name}", str(root / name), f"--{name}-sha256", "a" * 64]
            argv += [
                "--slides-uri",
                "mlflow-artifacts:/3/tiling/artifacts/slides.parquet",
            ]
            with (
                patch.object(
                    pilot, "execute", side_effect=AssertionError("execute called")
                ),
                patch.object(
                    pilot, "urlopen", side_effect=AssertionError("network called")
                ),
            ):
                result = pilot.main(argv)
            self.assertEqual(result["mode"], "plan")
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "cache").exists())

    def test_checksum_rejects_existing_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "cached"
            p.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                pilot.verify_file(p, hashlib.sha256(b"expected").hexdigest())

    def test_stream_caps_unknown_length_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "slide.zip"
            budget = pilot.ByteBudget(10)
            with (
                patch.object(pilot, "urlopen", return_value=Response(b"x" * 11)),
                self.assertRaises(pilot.ByteCapExceededError),
            ):
                pilot.download_bounded("https://example/a", p, 8, budget)
            self.assertFalse(p.exists())
            self.assertFalse(p.with_suffix(".zip.part").exists())
            self.assertEqual(budget.used, 8)

    def test_content_length_and_combined_caps(self):
        with tempfile.TemporaryDirectory() as temp:
            budget = pilot.ByteBudget(6)
            p = Path(temp) / "slide.zip"
            with (
                patch.object(pilot, "urlopen", return_value=Response(b"abc", 9)),
                self.assertRaises(pilot.ByteCapExceededError),
            ):
                pilot.download_bounded("https://example/a", p, 8, budget)
            with patch.object(pilot, "urlopen", return_value=Response(b"abc")):
                pilot.download_bounded("https://example/a", p, 8, budget)
            with (
                patch.object(pilot, "urlopen", return_value=Response(b"defg")),
                self.assertRaises(pilot.ByteCapExceededError),
            ):
                pilot.download_bounded(
                    "https://example/a", Path(temp) / "other.zip", 8, budget
                )
            self.assertEqual(budget.used, 6)

    def make_zip(self, path, entries):
        with zipfile.ZipFile(path, "w") as archive:
            for name, contents in entries:
                archive.writestr(name, contents)

    def test_zip_preserves_windows_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root / "slide.zip"
            self.make_zip(
                p,
                [
                    ("slide.mrxs", b"m"),
                    ("slide\\", b""),
                    ("slide\\Index.dat", b"i"),
                    ("slide\\Slidedat.ini", b"s"),
                ],
            )
            pilot.extract_safe(p, root / "extracted", 20, pilot.ByteBudget(20))
            self.assertEqual((root / "extracted/slide/Index.dat").read_bytes(), b"i")

    def test_zip_rejects_traversal_symlink_collisions_and_bombs(self):
        bad = ["../escape", "..\\escape", "/absolute", "C:\\absolute", "a/../../escape"]
        for name in bad:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_zip(root / "a.zip", [(name, b"x")])
                with self.assertRaises(ValueError):
                    pilot.extract_safe(
                        root / "a.zip", root / "dest", 10, pilot.ByteBudget(10)
                    )
                self.assertFalse((root / "dest").exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            for entries in [
                [("a\\b", b"1"), ("a/b", b"2")],
                [(info, b"target")],
                [("a", b"x" * 11)],
                [("a", b"1"), ("a/b", b"2")],
            ]:
                self.make_zip(root / "a.zip", entries)
                with self.assertRaises((ValueError, pilot.ByteCapExceededError)):
                    pilot.extract_safe(
                        root / "a.zip", root / "dest", 10, pilot.ByteBudget(10)
                    )
                self.assertFalse((root / "dest").exists())

    def test_target_identity_rejects_wrong_run_experiment_and_sha(self):
        run = {
            "info": {"run_id": "run", "experiment_id": "3", "status": "FINISHED"},
            "data": {"tags": [{"key": "mlflow.source.git.commit", "value": "b" * 40}]},
        }
        experiment = {"experiment_id": "3", "name": "MammaPrint"}
        pilot.check_identity(run, experiment, "run", "3", "MammaPrint")
        for key, value in [
            ("run_id", "other"),
            ("experiment_id", "7"),
            ("status", "RUNNING"),
        ]:
            changed = {**run, "info": {**run["info"], key: value}}
            with self.assertRaises(ValueError):
                pilot.check_identity(changed, experiment, "run", "3", "MammaPrint")
        with self.assertRaises(ValueError):
            pilot.check_identity(
                run, {**experiment, "name": "other"}, "run", "3", "MammaPrint"
            )

    def test_selected_split_audit_truthful_no_test_bags_or_patient_claim(self):
        assets = [
            {"slide_id": "a", "split": "train"},
            {"slide_id": "b", "split": "val"},
        ]
        result = pilot.selected_split_audit(
            assets, {"train": {"a"}, "val": {"b"}, "test": {"c"}}
        )
        self.assertEqual(
            result["scope"],
            "original artifact memberships; selected complete train/val bags only",
        )
        self.assertEqual(
            result["patient_audit"],
            "unverified: original patient/case mapping unavailable",
        )
        self.assertEqual(result["test_bags_downloaded_or_scored"], 0)
        with self.assertRaises(ValueError):
            pilot.selected_split_audit(
                assets, {"train": {"b"}, "val": {"a"}, "test": {"c"}}
            )
        with self.assertRaises(ValueError):
            pilot.selected_split_audit(
                assets, {"train": {"a"}, "val": {"b"}, "test": {"a"}}
            )


class LocalDatasetIntegrationTests(unittest.TestCase):
    def test_real_local_dataset_retains_complete_bag_and_original_uri_identity(self):
        # Uses the production dataset loader, and fails if it tries an HTTP download.
        import numpy as np
        import pandas as pd
        from omegaconf import OmegaConf

        from ml.concept_explainability.data import corpus_identity, iter_slides
        from ml.concept_explainability.model import SourceRun
        from ml.data.datasets.single_scale import SingleScaleDataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "bags"
            directory.mkdir()
            frame = pd.DataFrame(
                {
                    "x": np.arange(17),
                    "y": np.zeros(17, dtype=int),
                    "embedding": list(np.ones((17, 2560), dtype=np.float32)),
                }
            )
            frame.to_parquet(directory / "sample.parquet")
            pd.DataFrame(
                {"path": ["//mnt/data/project/sample.mrxs"], "type": ["a luminal"]}
            ).to_csv(root / "mapping.csv", index=False)
            original = {
                "3": {
                    "mpp": 1.94,
                    "tile_extent": 224,
                    "stride": 112,
                    "uris": {"train": "mlflow-artifacts:/3/original/artifacts/train"},
                }
            }
            local = {"3": {**original["3"], "uris": {"train": str(directory)}}}
            with patch(
                "requests.sessions.Session.request",
                side_effect=AssertionError("unexpected remote request"),
            ):
                dataset = SingleScaleDataset(
                    "train", local, root / "mapping.csv", "type"
                )
            source = SourceRun(
                "source",
                "b" * 40,
                OmegaConf.create(
                    {"dataset": {"levels": original}, "feature_dim": 2560}
                ),
                None,
                None,
                {"train": dataset},
            )
            slides = list(iter_slides(source, "train"))
            self.assertEqual(len(slides[0].embeddings), 17)
            self.assertEqual(slides[0].keys.row_index.tolist(), list(range(17)))
            self.assertEqual(
                corpus_identity(source)["train_uri"], original["3"]["uris"]["train"]
            )
            self.assertNotIn("test", source.datasets)


class ExecutePreparationRegressionTests(unittest.TestCase):
    def prepare_fixture(
        self, root, *, missing_label=False, empty_bag=False, wrong_rows=False
    ):
        import subprocess

        import numpy as np
        import pandas as pd
        from omegaconf import OmegaConf

        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pilot.REPO, text=True
        ).strip()
        uris = {
            split: f"mlflow-artifacts:/3/original/artifacts/{split}"
            for split in ("train", "val", "test")
        }
        config = OmegaConf.create(
            {
                "label_mode": "type",
                "feature_dim": 2560,
                "dataset": {
                    "levels": {
                        "3": {
                            "mpp": 1.94,
                            "tile_extent": 224,
                            "stride": 112,
                            "uris": uris,
                        }
                    }
                },
            }
        )
        config_path = root / "source.yaml"
        OmegaConf.save(config, config_path)
        checkpoint = root / "checkpoint"
        checkpoint.write_bytes(b"not loaded: model computation prohibited by test")
        assets, rows = [], []
        for split in ("train", "val"):
            for index in range(2):
                sid = f"{split}{index}"
                count = 0 if empty_bag and sid == "train1" else 17
                path = root / f"{sid}.parquet"
                pd.DataFrame(
                    {
                        "x": np.arange(count),
                        "y": np.zeros(count, dtype=int),
                        "embedding": list(np.ones((count, 2560), dtype=np.float32)),
                    }
                ).to_parquet(path)
                assets.append(
                    {
                        "slide_id": sid,
                        "split": split,
                        "source_uri": uris[split] + f"/{sid}.parquet",
                        "local_path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                        "rows": 18 if wrong_rows and sid == "val1" else 17,
                    }
                )
                rows.append(
                    {
                        "path": f"//mnt/data/project/{sid}.mrxs",
                        "type": None
                        if missing_label and sid == "val1"
                        else "a luminal",
                    }
                )
        metadata = root / "slides.parquet"
        pd.DataFrame(rows).to_parquet(metadata)
        manifest = root / "assets.json"
        manifest.write_text(json.dumps(assets))
        argv = [
            "--tracking-uri",
            "https://mlflow.example",
            "--wsi-base-url",
            "https://wsi.example/v3",
            "--source-run-id",
            "run",
            "--expected-experiment-id",
            "3",
            "--expected-experiment-name",
            "MammaPrint",
            "--assets-manifest",
            str(manifest),
            "--output-root",
            str(root / "output"),
            "--cache-root",
            str(root / "cache"),
            "--slides-uri",
            "mlflow-artifacts:/3/tiling/artifacts/slides.parquet",
            "--execute",
        ]
        for name, path in (
            ("source-config", config_path),
            ("checkpoint", checkpoint),
            ("slides", metadata),
        ):
            argv += [
                f"--{name}",
                str(path),
                f"--{name}-sha256",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
        args = pilot.parser().parse_args(argv)
        pilot.validate_args(args)
        run = {
            "info": {"run_id": "run", "experiment_id": "3", "status": "FINISHED"},
            "data": {"tags": [{"key": "mlflow.source.git.commit", "value": revision}]},
        }

        def metadata_get(base, route, **params):
            if route.endswith("runs/get"):
                return {"run": run}
            if route.endswith("experiments/get"):
                return {"experiment": {"experiment_id": "3", "name": "MammaPrint"}}
            raise AssertionError("Unexpected metadata request")

        def memberships(base, uri):
            split = uri.rsplit("/", 1)[-1]
            return {f"{split}0", f"{split}1"}

        return args, assets, metadata_get, memberships

    def run_preparation(self, root, *, skip_wsi=False, **changes):
        from ml.concept_explainability import model

        args, assets, metadata_get, memberships = self.prepare_fixture(root, **changes)
        args.skip_wsi = skip_wsi
        # wraps preserves the real keyword-only signature AND runs the real Git guard.
        with (
            patch.object(
                model, "verify_training_revision", wraps=model.verify_training_revision
            ) as guard,
            patch.object(pilot, "json_get", side_effect=metadata_get),
            patch.object(pilot, "membership", side_effect=memberships),
            patch.object(
                pilot,
                "download_bounded",
                side_effect=AssertionError("WSI body forbidden"),
            ) as download,
            patch(
                "ml.concept_explainability.evaluation.fit_dictionary",
                side_effect=RuntimeError("preparation complete"),
            ),
            patch(
                "requests.sessions.Session.request",
                side_effect=AssertionError("HTTP forbidden"),
            ),
        ):
            try:
                pilot.execute(args, assets)
            finally:
                self.assertEqual(guard.call_count, 1)
                self.assertIn("config", guard.call_args.kwargs)
                download.assert_not_called()

    def test_execute_preparation_uses_real_keyword_only_source_guard(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(RuntimeError, "preparation complete"),
        ):
            self.run_preparation(Path(temp), skip_wsi=True)

    def test_execute_rejects_selected_bag_dropped_for_missing_label(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "roster"),
        ):
            self.run_preparation(Path(temp), missing_label=True)

    def test_execute_rejects_empty_selected_bag_before_download_or_fit(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "roster|nonzero"),
        ):
            self.run_preparation(Path(temp), empty_bag=True)

    def test_execute_rejects_wrong_inspected_complete_row_count(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "row count"),
        ):
            self.run_preparation(Path(temp), wrong_rows=True)


if __name__ == "__main__":
    unittest.main()
