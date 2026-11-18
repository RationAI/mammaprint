"""Split metadata isolation and the single-process fit/explain contract."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from mlflow.entities import FileInfo
from omegaconf import OmegaConf

from ml.concept_explainability.data import audit_splits
from ml.concept_explainability.model import SourceRun
from ml.concepts import run


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = self.root / "mapping.csv"
        self.rows = pd.DataFrame(
            {
                "path": ["train.svs", "val.svs", "test.svs", "unlabeled.svs"],
                "type": ["a luminal", "b luminal", "a luminal", None],
                "mammaprint_index": [0.1, 0.2, 0.3, 0.4],
                "patient_id": ["a", "b", "c", "d"],
            }
        )
        self.rows.to_csv(self.mapping, index=False)
        self.source = SourceRun(
            "source",
            None,
            OmegaConf.create(
                {
                    "datamodule": {
                        split: {
                            "_target_": "ml.data.SingleScaleDataset",
                            "split": split,
                            "data_mapping": str(self.mapping),
                            "label_mode": "type",
                            "levels": {"3": {"uris": {split: f"runs:/source/{split}"}}},
                        }
                        for split in SPLITS
                    },
                }
            ),
            None,
            None,
        )
        self.listings = {
            f"runs:/source/{s}": [
                FileInfo(f"{s}/{s}.parquet", False, 0),
                FileInfo(f"{s}/unlabeled.parquet", False, 99),
                FileInfo(f"{s}/nested", True, None),
                FileInfo(f"{s}/readme.txt", False, 10),
            ]
            for s in SPLITS
        }

    def audit(self):
        with (
            patch(
                "mlflow.artifacts.list_artifacts",
                side_effect=lambda *, artifact_uri: self.listings[artifact_uri],
            ) as listing,
            patch(
                "ml.concept_explainability.data.instantiate",
                side_effect=AssertionError("dataset instantiated"),
            ),
            patch(
                "ml.concept_explainability.data.source_dataset",
                side_effect=AssertionError("dataset resolved"),
            ),
            patch(
                "ml.data.datasets._sources.download_level_sources",
                side_effect=AssertionError("download"),
            ),
            patch(
                "pandas.read_parquet", side_effect=AssertionError("parquet body read")
            ),
            patch(
                "pyarrow.parquet.read_metadata",
                side_effect=AssertionError("parquet metadata read"),
            ),
        ):
            result = audit_splits(self.source)
        return result, listing

    def test_metadata_only_membership_preserves_objective_and_flat_files(self):
        result, listing = self.audit()
        self.assertEqual(listing.call_count, 3)
        self.assertTrue(result["slide_disjoint"])
        self.assertEqual(result["patient_key"], "patient_id")
        for split in SPLITS:
            evidence = result["splits"][split]
            self.assertEqual(evidence["membership_source"], "artifact_listing")
            self.assertEqual(evidence["n_slides"], 1)
            self.assertIn("unverified", evidence["empty_bags"])
        self.assertEqual(self.source.datasets, {})

    def test_original_per_split_objective_controls_usable_labels(self):
        self.source.config.datamodule.test.label_mode = "index"
        self.listings["runs:/source/test"] = [
            FileInfo("test/unlabeled.parquet", False, 0)
        ]
        result, _ = self.audit()
        self.assertEqual(result["splits"]["test"]["n_slides"], 1)
        # BOTH drops a slide missing either original objective column.
        self.source.config.datamodule.test.label_mode = "both"
        with self.assertRaisesRegex(ValueError, "Empty usable test"):
            self.audit()

    def test_omitted_label_mode_matches_native_type_label_roster(self):
        from ml.data.datasets._sources import load_labeled_slides
        from ml.data.datasets.labels import LabelMode

        expected = load_labeled_slides(self.mapping, LabelMode.TYPE)
        self.assertNotIn("unlabeled", set(expected["name"]))
        for target in ("ml.data.SingleScaleDataset", "ml.data.SpatialScaleDataset"):
            with self.subTest(target=target):
                for split in SPLITS:
                    config = self.source.config.datamodule[split]
                    config._target_ = target
                    config.pop("label_mode", None)
                result, listing = self.audit()
                self.assertEqual(listing.call_count, 3)
                for split in SPLITS:
                    # The fixture lists one labeled and one unlabeled slide in
                    # each physical split. Match production's default objective.
                    roster = expected[expected["name"].isin({split, "unlabeled"})]
                    self.assertEqual(set(roster["name"]), {split})
                    self.assertEqual(result["splits"][split]["n_slides"], len(roster))
                    self.assertEqual(
                        result["splits"][split]["membership_source"], "artifact_listing"
                    )
                self.assertEqual(self.source.datasets, {})

    def test_explicit_none_label_mode_remains_invalid(self):
        self.source.config.datamodule.train.label_mode = None
        with self.assertRaisesRegex(ValueError, "None is not a valid LabelMode"):
            self.audit()

    def test_listed_empty_bags_are_conservatively_checked_for_leakage(self):
        self.listings["runs:/source/test"].append(
            FileInfo("test/train.parquet", False, 0)
        )
        with self.assertRaisesRegex(ValueError, "Slide leakage between train and test"):
            self.audit()

    def test_patient_leakage_is_rejected_without_bags(self):
        self.rows.loc[2, "patient_id"] = "a"
        self.rows.to_csv(self.mapping, index=False)
        with self.assertRaisesRegex(ValueError, "Patient/case leakage.*train and test"):
            self.audit()

    def test_all_levels_are_intersected_and_split_mapping_is_authority(self):
        self.source.config.datamodule.train.levels["4"] = {
            "uris": {"train": "runs:/other/train"}
        }
        self.listings["runs:/other/train"] = [
            FileInfo("other/train/train.parquet", False, 1)
        ]
        # This labeled file is in only one train level: it must not leak to val.
        self.listings["runs:/source/train"].append(
            FileInfo("train/val.parquet", False, 1)
        )
        result, listing = self.audit()
        self.assertEqual(listing.call_count, 4)
        self.assertEqual(result["splits"]["train"]["n_slides"], 1)

    def test_duplicate_usable_mapping_rows_are_rejected(self):
        pd.concat([self.rows, self.rows.iloc[:1]]).to_csv(self.mapping, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate slide"):
            self.audit()

    def test_cached_sources_remain_offline(self):
        for split, patient in zip(SPLITS, ("a", "b", "c"), strict=True):
            self.source.datasets[split] = SimpleNamespace(
                slides=pd.DataFrame({"name": [split], "case_id": [patient]}),
                empty_slides=(),
            )
        with (
            patch(
                "mlflow.artifacts.list_artifacts",
                side_effect=AssertionError("remote list"),
            ),
            patch(
                "ml.concept_explainability.data.instantiate",
                side_effect=AssertionError("instantiate"),
            ),
        ):
            result = audit_splits(self.source)
        self.assertEqual(result["patient_key"], "case_id")
        self.assertEqual(
            {e["membership_source"] for e in result["splits"].values()},
            {"cached_dataset"},
        )


def workflow_config(output, *, include_test=False, metric=False):
    config = OmegaConf.load(ROOT / "configs/concepts.yaml")
    config.action = "fit_explain"
    config.source_run_id = "source"
    config.output_dir = str(output)
    config.device = "cpu"
    config.fixture_mode = True
    config.log_to_mlflow = False
    config.include_test = include_test
    if metric:
        config.fit.metric = {
            "enabled": True,
            "ridge": 0.3,
            "n_tiles": 20000,
            "attention_weight": 1.0,
        }
    return config


class CombinedWorkflowTests(unittest.TestCase):
    def test_deployed_config_runs_native_metric_with_one_source_and_no_test_reads(self):
        from hydra import compose, initialize_config_dir
        from PIL import Image
        from test_concept_evaluation import fixture_source, local_download
        from test_concept_metric import metric_source
        from test_concept_sources import BASE

        from ml.concept_explainability.model import load_source_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = fixture_source(root)
            native, _ = metric_source(root, 2560)
            source.config.ml = native.config.ml
            source.config.datamodule["_target_"] = "ml.data.DataModule"
            card = source.config.dataset.levels.pop("3")
            source.config.dataset.levels["0"] = card
            for split in SPLITS:
                for i in range(2):
                    Image.new("RGB", (512, 512), (100, 50, 150)).save(
                        root / f"{split}{i}.tiff"
                    )
            config_path = root / "resolved.yaml"
            OmegaConf.save(source.config, config_path)
            checkpoint_before = native.checkpoint_path.read_bytes()
            loaded = []

            def load(*args, **kwargs):
                recovered = load_source_run(*args, **kwargs)
                loaded.append(recovered)
                return recovered

            read_parquet = pd.read_parquet

            def read(path, *args, **kwargs):
                self.assertNotEqual(
                    Path(path).parent, root / "test", "default workflow read a test bag"
                )
                return read_parquet(path, *args, **kwargs)

            with initialize_config_dir(
                config_dir=str(ROOT / "configs"), version_base=None
            ):
                config = compose(
                    config_name="concepts_deployed",
                    overrides=[
                        "source_run_id=source",
                        f"output_dir={root / 'output'}",
                        "fixture_mode=true",
                        "log_to_mlflow=false",
                        "device=cpu",
                        "fit.n_components=2",
                        "fit.nmf.batch_size=32",
                        "fit.nmf.shuffle_buffer_size=32",
                        "fit.nmf.nnls_iter=10",
                    ],
                )
            self.assertEqual(config.action, "fit_explain")
            self.assertIs(config.include_test, False)
            with (
                patch("mlflow.MlflowClient") as client,
                patch("ml.concepts.load_source_run", side_effect=load) as source_load,
                patch(
                    "ml.data.datasets.single_scale.download_level_sources",
                    side_effect=local_download,
                ) as download,
                patch("pandas.read_parquet", side_effect=read),
            ):
                client.return_value.get_run.return_value = SimpleNamespace(
                    data=SimpleNamespace(tags={"mlflow.source.git.commit": BASE})
                )
                client.return_value.download_artifacts.side_effect = [
                    str(config_path),
                    str(native.checkpoint_path),
                ]
                output = run(config)
            source_load.assert_called_once()
            self.assertEqual(client.return_value.download_artifacts.call_count, 2)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(set(loaded[0].datasets), {"train", "val"})
            self.assertEqual(download.call_count, 2)
            self.assertEqual(
                {
                    Path(next(iter(call.args[0].values()))).name
                    for call in download.call_args_list
                },
                {"train", "val"},
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["metric_enabled"])
            self.assertEqual(manifest["metric"]["sample"]["population_tiles"], 32)
            self.assertEqual(
                manifest["split_audit"]["splits"]["test"]["membership_source"],
                "artifact_listing",
            )
            self.assertTrue((output / "dictionary/transform.npy").is_file())
            self.assertTrue((output / "val/naming_k2.html").is_file())
            self.assertFalse((output / "test").exists())
            self.assertEqual(native.checkpoint_path.read_bytes(), checkpoint_before)
            for stage in ("dictionary", "val"):
                detail = json.loads((output / stage / "manifest.json").read_text())
                self.assertEqual(
                    detail["implementation_sha"], manifest["implementation_sha"]
                )
            with (
                patch(
                    "ml.concepts.load_source_run",
                    side_effect=AssertionError("reuse accessed source"),
                ),
                self.assertRaisesRegex(ValueError, "previous results"),
            ):
                run(config)

    def stages(self, root, *, include_test=False, log=False):
        config = workflow_config(root / "output", include_test=include_test)
        config.log_to_mlflow = log
        source = SourceRun(
            "source", "a" * 40, OmegaConf.create({"graph": "fixture"}), None, None
        )
        events = []

        def fit(actual_source, settings, output, **kwargs):
            self.assertIs(actual_source, source)
            events.append(("fit", "train"))
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(
                json.dumps(
                    {"dictionary_id": "d", "n_components": 16, "fixture_only": True}
                )
            )

        def evaluate(actual_source, dictionary, split, output, **kwargs):
            self.assertIs(actual_source, source)
            self.assertEqual(dictionary, (root / "output/dictionary").resolve())
            events.append(("evaluate", split))
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(
                json.dumps({"dictionary_id": "d", "split": split, "fixture_only": True})
            )

        def report(actual_source, dictionary, evaluation, output, **kwargs):
            self.assertIs(actual_source, source)
            self.assertEqual(evaluation, output)
            (output / "naming_k16.html").write_text("fixture report")
            (output / "report_manifest.json").write_text(
                (output / "manifest.json").read_text()
            )
            return output / "naming_k16.html"

        return config, source, events, fit, evaluate, report

    def test_combined_fits_once_and_explains_only_requested_splits(self):
        for include_test in (False, True):
            with (
                self.subTest(include_test=include_test),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                config, source, events, fit, evaluate, report = self.stages(
                    root, include_test=include_test
                )
                with (
                    patch("ml.concepts.load_source_run", return_value=source) as load,
                    patch(
                        "ml.concepts.audit_splits",
                        return_value={"slide_disjoint": True},
                    ),
                    patch("ml.concepts.fit_dictionary", side_effect=fit),
                    patch("ml.concepts.evaluate_concepts", side_effect=evaluate),
                    patch("ml.concepts.build_report", side_effect=report),
                ):
                    output = run(config)
                load.assert_called_once()
                self.assertTrue(load.call_args.kwargs["download_weights"])
                self.assertEqual(
                    events,
                    [("fit", "train"), ("evaluate", "val")]
                    + ([("evaluate", "test")] if include_test else []),
                )
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["action"], "fit_explain")
                self.assertEqual(manifest["dictionary_fit_device"], "cpu")
                self.assertEqual(manifest["explanation_device"], "cpu")
                self.assertEqual(
                    set(manifest["stages"]),
                    {"dictionary", "val", "test"}
                    if include_test
                    else {"dictionary", "val"},
                )
                self.assertEqual((output / "test").exists(), include_test)

    def test_single_fresh_destination_run_publishes_all_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, _events, fit, evaluate, report = self.stages(root, log=True)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("ml.concepts.load_source_run", return_value=source),
                patch("ml.concepts.audit_splits", return_value={}),
                patch("ml.concepts.fit_dictionary", side_effect=fit),
                patch("ml.concepts.evaluate_concepts", side_effect=evaluate),
                patch("ml.concepts.build_report", side_effect=report),
                patch("mlflow.active_run", return_value=None),
                patch("mlflow.MlflowClient") as client,
                patch(
                    "mlflow.start_run", return_value=contextlib.nullcontext()
                ) as start,
                patch("mlflow.log_artifacts") as publish,
            ):
                client.return_value.get_experiment_by_name.return_value = (
                    SimpleNamespace(experiment_id="new-exp")
                )
                destination = SimpleNamespace(
                    info=SimpleNamespace(run_id="fresh-destination")
                )
                start.return_value = contextlib.nullcontext(destination)
                start.return_value.info = destination.info
                output = run(config)
            start.assert_called_once()
            publish.assert_called_once_with(str(output))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["publication"]["run_id"], "fresh-destination")

    def test_unsupported_metric_source_fails_before_audit_or_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = workflow_config(root / "output", metric=True)
            config.log_to_mlflow = True
            source = SourceRun(
                "source",
                "a" * 40,
                OmegaConf.create({"label_mode": "both"}),
                None,
                None,
            )
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("ml.concepts.load_source_run", return_value=source),
                patch(
                    "ml.concepts.audit_splits",
                    side_effect=AssertionError("audit before compatibility"),
                ),
                patch(
                    "mlflow.MlflowClient",
                    side_effect=AssertionError("new publication before compatibility"),
                ),
                self.assertRaisesRegex(
                    ValueError, "metric.*label_mode|metric requires label_mode"
                ),
            ):
                run(config)
            self.assertFalse((root / "output").exists())

    def test_cuda_unavailable_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, *_ = self.stages(root)
            config.device = "cuda"
            with (
                patch("ml.concepts.load_source_run", return_value=source),
                patch("ml.concepts.audit_splits", return_value={}),
                patch("torch.cuda.is_available", return_value=False),
                patch(
                    "ml.concepts.fit_dictionary",
                    side_effect=AssertionError("fit before device validation"),
                ),
                self.assertRaisesRegex(ValueError, "cuda"),
            ):
                run(config)

    def test_inherited_run_guard_precedes_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config = workflow_config(Path(directory) / "output")
            config.log_to_mlflow = True
            with (
                patch.dict("os.environ", {"MLFLOW_RUN_ID": "training"}),
                patch(
                    "ml.concepts.load_source_run",
                    side_effect=AssertionError("source access"),
                ),
                self.assertRaisesRegex(RuntimeError, "MLFLOW_RUN_ID"),
            ):
                run(config)

    def test_nonboolean_test_opt_in_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = workflow_config(Path(directory) / "output")
            config.include_test = "false"
            with (
                patch(
                    "ml.concepts.load_source_run",
                    side_effect=AssertionError("source access"),
                ),
                self.assertRaisesRegex(ValueError, "include_test.*boolean"),
            ):
                run(config)


if __name__ == "__main__":
    unittest.main()
