from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.explainability.dataset_mask_lineage import MaskSource, MaskSources
from ml.explainability.dataset_mask_report import (
    build_layer_manifest,
    layer_manifest_bytes,
    patch_report_layers,
)
from scripts.ml.prepare_dataset_mask_previews import prepare
from tests.test_dataset_mask_reports import HOST, SOURCE, report, slide_table
from tests.test_tile_removal_reports import _FakeMlflow


RUN = "1" * 32
TRACKING = "https://mlflow.example.test"
BATCH = "mask-preview"
FOLDER = f"report_backfills/dataset_masks/previews/{BATCH}"


class DatasetMaskPreviewWorkerTest(unittest.TestCase):
    def setUp(self):
        self.client = _FakeMlflow()
        self.original = self.client.artifacts["report/report.html"]
        self.digest = hashlib.sha256(self.original).hexdigest()
        self.client.artifacts["explanations/manifest.json"] = json.dumps(
            {"explanation_run_id": RUN, "explanation_experiment_id": "3"}
        ).encode()
        self.client.artifacts["explanations/slides.parquet"] = b"mock slide table"

    def mock_preview(self, args):
        self.assertNotIn("--apply", args)
        self.assertEqual(
            args[args.index("--mask-export-viewer-prefix") + 1],
            f"secure_mlflow/3/{RUN}/artifacts/{FOLDER}/dataset-mask-assets",
        )
        output = Path(args[args.index("--mask-export-dir") + 1])
        output.mkdir()
        tiff = b"mock synthetic TIFF"
        digest = hashlib.sha256(tiff).hexdigest()
        (output / f"{digest}.ome.tif").write_bytes(tiff)
        preview = Path(args[args.index("--preview") + 1])
        preview.write_bytes(b"mock preview")
        layers = Path(args[args.index("--write-layer-manifest") + 1])
        layers.write_text(json.dumps({"missing": {"slide": {"tissue": "missing"}}}))
        print(
            json.dumps(
                {
                    "original_sha256": self.digest,
                    "updated_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
                    "slides_with_masks": 1,
                }
            )
        )
        return 0

    def test_only_new_preview_artifacts_are_staged(self):
        with (
            patch(
                "scripts.ml.prepare_dataset_mask_previews.preview_report",
                side_effect=self.mock_preview,
            ),
            redirect_stdout(io.StringIO()),
        ):
            receipt = prepare(
                self.client, TRACKING, "3", RUN, self.digest, BATCH, stage=True
            )
        self.assertEqual(receipt["converted_files"], 1)
        self.assertEqual(receipt["missing_reasons"], {"missing": 1})
        self.assertEqual(self.client.artifacts["report/report.html"], self.original)
        self.assertEqual(len(self.client.writes), 4)
        self.assertTrue(
            all(path.startswith(FOLDER + "/") for path in self.client.writes)
        )

    def test_report_changed_after_preflight_stops_before_prepare(self):
        self.client.artifacts["report/report.html"] = b"changed HTML"
        with (
            patch("scripts.ml.prepare_dataset_mask_previews.preview_report") as preview,
            self.assertRaisesRegex(ValueError, "changed after"),
        ):
            prepare(self.client, TRACKING, "3", RUN, self.digest, BATCH, stage=True)
        preview.assert_not_called()
        self.assertEqual(self.client.writes, [])

    def test_no_staging_by_default_path(self):
        with (
            patch(
                "scripts.ml.prepare_dataset_mask_previews.preview_report",
                side_effect=self.mock_preview,
            ),
            redirect_stdout(io.StringIO()),
        ):
            receipt = prepare(
                self.client, TRACKING, "3", RUN, self.digest, BATCH, stage=False
            )
        self.assertIsNone(receipt["stage"])
        self.assertEqual(self.client.writes, [])

    def test_local_output_requires_new_conversions_to_be_staged(self):
        with (
            tempfile.TemporaryDirectory() as work,
            patch(
                "scripts.ml.prepare_dataset_mask_previews.preview_report",
                side_effect=self.mock_preview,
            ),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "links new converted assets"),
        ):
            prepare(
                self.client,
                TRACKING,
                "3",
                RUN,
                self.digest,
                BATCH,
                stage=False,
                output_dir=Path(work),
            )
        self.assertEqual(self.client.writes, [])


class DatasetMaskPreviewReuseTest(unittest.TestCase):
    donor = "9" * 32
    donor_batch = "verified-donor"
    donor_folder = f"report_backfills/dataset_masks/previews/{donor_batch}"
    tissue = MaskSource("/mnt/project/masks/tissue", "tissue gate", "donor config")
    blur = MaskSource("/mnt/project/masks/qc", "blur gate", "donor config")
    epithelium = MaskSource("/mnt/project/masks/epi", "epi gate", "target config")

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.local_target = _FakeMlflow()
        self.local_donor = _FakeMlflow()
        self.client = unittest.mock.Mock()
        self.client.get_run.side_effect = lambda run: self._client(run).get_run(run)
        self.client.download_artifacts.side_effect = self.download
        self.client.list_artifacts.side_effect = self.list_artifacts
        self.directories = set()
        self.slides = slide_table()
        self.original = report()
        self.original_hash = hashlib.sha256(self.original).hexdigest()
        self.local_target.artifacts.update(
            {
                "report/report.html": self.original,
                "explanations/manifest.json": json.dumps(SOURCE).encode(),
                "explanations/slides.parquet": self.slides,
            }
        )
        self.donor_provenance = {**SOURCE, "explanation_run_id": self.donor}
        self.donor_original = self.original.replace(RUN.encode(), self.donor.encode())
        self.donor_sources = MaskSources(
            "donor dataset", {"tissue": self.tissue, "blur": self.blur}
        )
        self.target_sources = MaskSources(
            "target dataset",
            {
                "tissue": MaskSource(
                    self.tissue.location, "different tissue gate", "target config"
                ),
                "blur": self.blur,
            },
        )
        tiff = b"mock donor export"
        digest = hashlib.sha256(tiff).hexdigest()
        self.staged = f"{self.donor_folder}/dataset-mask-assets/{digest}.ome.tif"
        self.donor_assets = {
            "mock": {
                "blur": {
                    "viewer_path": f"secure_mlflow/3/{self.donor}/artifacts/{self.staged}",
                    "source": f"{self.blur.location}/Piqe_piqe_median_activity_mask_mock-case.tiff",
                    "sha256": digest,
                },
                "tissue": {
                    "viewer_path": "project/masks/tissue/mock-case.tiff",
                    "source": f"{self.tissue.location}/mock-case.tiff",
                    "sha256": "f" * 64,
                },
            }
        }
        donor_layers = build_layer_manifest(
            self.donor_provenance,
            self.slides,
            dataset_uri=self.donor_sources.dataset_uri,
            assets=self.donor_assets,
            missing={},
            selections={
                kind: {"selection": v.selection, "evidence": v.evidence}
                for kind, v in self.donor_sources.layers.items()
            },
        )
        rendered, _ = patch_report_layers(
            self.donor_original,
            self.donor_provenance,
            self.slides,
            donor_layers,
            xopat_ui_uri=HOST,
            artifact_root="explanations",
            artifact_mount="secure_mlflow",
        )
        donor_bytes = layer_manifest_bytes(donor_layers)
        self.donor_receipt = {
            "stage": self.donor_folder,
            "status": "preview",
            "run_id": self.donor,
            "explanation_run_id": self.donor,
            "original_sha256": hashlib.sha256(self.donor_original).hexdigest(),
            "updated_sha256": hashlib.sha256(rendered).hexdigest(),
            "bundle_sha256": hashlib.sha256(donor_bytes).hexdigest(),
        }
        self.local_donor.artifacts.update(
            {
                "report/report.html": self.donor_original,
                "explanations/manifest.json": json.dumps(
                    self.donor_provenance
                ).encode(),
                "explanations/slides.parquet": self.slides,
                f"{self.donor_folder}/layers.json": donor_bytes,
                f"{self.donor_folder}/preview.html": rendered,
                f"{self.donor_folder}/receipt.json": json.dumps(
                    self.donor_receipt
                ).encode(),
                self.staged: tiff,
            }
        )

    def _client(self, run):
        self.assertIn(run, {RUN, self.donor})
        return self.local_donor if run == self.donor else self.local_target

    def download(self, run, artifact, directory):
        return self._client(run).download_artifacts(run, artifact, directory)

    def list_artifacts(self, run, prefix):
        files = self._client(run).list_artifacts(run, prefix)
        for item in files:
            item.is_dir = item.path in self.directories
        return files

    def resolve(self, client, provenance, config):
        self.assertIs(client, self.client)
        self.assertIsNone(config)
        return (
            self.donor_sources
            if provenance["explanation_run_id"] == self.donor
            else self.target_sources
        )

    def prepare_with_reuse(self, prepared=None):
        with (
            patch(
                "scripts.ml.prepare_dataset_mask_previews.resolve_mask_sources",
                side_effect=self.resolve,
            ),
            patch(
                "scripts.ml.prepare_dataset_mask_previews.prepare_mask_assets",
                side_effect=prepared,
            ) as asset_preparation,
            redirect_stdout(io.StringIO()),
        ):
            result = prepare(
                self.client,
                TRACKING,
                "3",
                RUN,
                self.original_hash,
                BATCH,
                stage=False,
                reuse=(self.donor, self.donor_batch),
                output_dir=self.root,
            )
        self.client.log_artifact.assert_not_called()
        self.assertEqual(
            self.local_target.artifacts["report/report.html"], self.original
        )
        return result, asset_preparation

    def test_exact_shared_layers_reuse_without_opening_source_slides(self):
        result, asset_preparation = self.prepare_with_reuse()
        asset_preparation.assert_not_called()
        self.assertEqual(result["layers_by_type"], {"tissue": 1, "blur": 1})
        self.assertEqual(result["patched_links"], 2)
        self.assertIsNone(result["stage"])
        self.assertEqual(result["converted_files"], 0)
        self.assertEqual(result["reuse_layer_run"], self.donor)
        saved = self.root / RUN
        manifest = json.loads((saved / "layers.json").read_bytes())
        self.assertEqual(manifest["assets"], self.donor_assets)
        self.assertEqual(manifest["report_identity"]["explanation_run_id"], RUN)
        self.assertEqual(manifest["dataset_uri"], "target dataset")
        self.assertEqual(manifest["selections"]["tissue"]["evidence"], "target config")
        self.assertTrue((saved / "preview.html").is_file())
        self.assertEqual(json.loads((saved / "receipt.json").read_bytes()), result)

    def test_target_only_epithelium_is_prepared(self):
        self.target_sources.layers["epithelium"] = self.epithelium

        def epithelium(client, sources, paths, **kwargs):
            self.assertIs(client, self.client)
            self.assertEqual(sources, {"epithelium": self.epithelium})
            self.assertEqual(paths, {"mock": "/mnt/project/slides/mock-case.mrxs"})
            return {
                "mock": {
                    "epithelium": {
                        "viewer_path": "project/masks/epi/mock-case.tiff",
                        "source": f"{self.epithelium.location}/mock-case.tiff",
                        "sha256": "e" * 64,
                    }
                }
            }, {}

        result, asset_preparation = self.prepare_with_reuse(prepared=epithelium)
        asset_preparation.assert_called_once()
        self.assertEqual(
            result["layers_by_type"], {"tissue": 1, "blur": 1, "epithelium": 1}
        )

    def test_stale_donor_is_rejected(self):
        self.local_donor.artifacts[f"{self.donor_folder}/preview.html"] = b"tampered"
        with self.assertRaisesRegex(ValueError, "Donor preview receipt"):
            self.prepare_with_reuse()
        self.client.log_artifact.assert_not_called()

    def test_absent_donor_export_is_rejected(self):
        del self.local_donor.artifacts[self.staged]
        with self.assertRaisesRegex(ValueError, "pinned export inventory"):
            self.prepare_with_reuse()

    def test_directory_cannot_stand_in_for_donor_export(self):
        self.directories.add(self.staged)
        with self.assertRaisesRegex(ValueError, "pinned export inventory"):
            self.prepare_with_reuse()

    def test_target_with_other_source_is_rejected(self):
        self.target_sources.layers["tissue"] = MaskSource(
            "/mnt/elsewhere", "tissue", "config"
        )
        with self.assertRaisesRegex(ValueError, "source locations differ"):
            self.prepare_with_reuse()
