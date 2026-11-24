from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.ml.prepare_dataset_mask_previews import prepare
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
