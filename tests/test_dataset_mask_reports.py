from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from html import escape, unescape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import yaml

from ml.explain_report import main as build_report
from ml.explainability.dataset_mask_report import (
    build_layer_manifest,
    patch_report_layers,
)
from ml.explainability.tile_removal_report import sha256
from scripts.ml.backfill_dataset_mask_reports import main as backfill


RUN = "1" * 32
HOST = "https://viewer.example.test"
ROOT = f"secure_mlflow/3/{RUN}/artifacts/explanations"
WSI = "project/slides/mock-case.mrxs"
SOURCE = {
    "explanation_run_id": RUN,
    "source_run_id": "2" * 32,
    "checkpoint_sha256": "3" * 64,
    "split": "test",
    "level": "l2",
    "pathologist_preview": {"xopat_ui_uri": HOST},
}
SELECTION = "mean cancer probability > 0.5"
MASK = "secure_mlflow/2/" + "4" * 32 + "/artifacts/heatmaps/mock-case.tiff"
SESSION = {
    "params": {"keep": "this"},
    "data": [WSI, f"{ROOT}/masks/attention/mammaprint_index/mock_positive.ome.tif"],
    "background": [{"dataReference": 0, "lossless": False, "protocol": "custom"}],
    "visualizations": [
        {
            "name": "mock",
            "lossless": True,
            "shaders": {
                "layer_shader_0": {
                    "name": "Attention weight",
                    "type": "heatmap",
                    "visible": True,
                    "dataReferences": [1],
                    "fixed": False,
                    "params": {"threshold": {"default": 1}},
                },
            },
        },
    ],
}


def viewer_link(config, *, host=HOST):
    fragment = quote(json.dumps(config), safe="")
    return host + "/xopat/redirect.php#" + fragment


def report():
    link = escape(viewer_link(SESSION), quote=True)
    unrelated = viewer_link(
        {**SESSION, "data": [WSI, "somewhere/tile-removal-review.tiff"]}
    )
    return (
        "<!doctype html>\r\n<html><head><title>Test Report</title></head>"
        "<body><h1>MammaPrint tile explanations</h1><p>Index: 0.1212</p>"
        f'<a title="href=unrelated" href="{link}">Slide image</a>'
        f"<a href='{link}'>Open viewer</a>"
        f'<a href="{unrelated}">Other review</a>'
        "<section>Previous report additions</section></body></html>"
    ).encode()


def slide_table(wsi=WSI):
    out = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "slide_id": "mock",
                    "slide_path": "/mnt/" + wsi,
                    "status": "ok",
                    "overlay_status": "ok",
                },
                {
                    "slide_id": "failed",
                    "slide_path": "/mnt/project/failed.mrxs",
                    "status": "error",
                    "overlay_status": "unavailable",
                },
            ]
        ),
        out,
    )
    return out.getvalue()


def layers(slides):
    return build_layer_manifest(
        SOURCE,
        slides,
        dataset_uri="mock-dataset",
        assets={
            "mock": {
                "cancer": {
                    "viewer_path": MASK,
                    "source": "mock-cancer-source",
                    "sha256": "5" * 64,
                }
            }
        },
        missing={"mock": {"folding": "file is unavailable"}},
        selections={"cancer": {"selection": SELECTION, "evidence": "tiling config"}},
    )


class DatasetMaskReportsTests(unittest.TestCase):
    def setUp(self):
        self.slides = slide_table()
        self.layers = layers(self.slides)
        self.original = report()

    def patch(self, data, manifest=None, provenance=None):
        return patch_report_layers(
            data,
            provenance or SOURCE,
            self.slides,
            manifest or self.layers,
            xopat_ui_uri=HOST,
            artifact_root="explanations",
            artifact_mount="secure_mlflow",
        )

    def test_both_links_preserve_original_session_and_add_hidden_layer(self):
        updated, summary = self.patch(self.original)
        urls = [
            unescape(url)
            for url in re.findall(
                r"""href=["'](https://viewer\.example\.test/xopat/redirect\.php#[^"']+)""",
                updated.decode(),
            )
        ]
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls[0], urls[1])
        self.assertEqual(
            urls[2],
            viewer_link(
                {**SESSION, "data": [WSI, "somewhere/tile-removal-review.tiff"]}
            ),
        )
        session = json.loads(unquote(urlsplit(urls[0]).fragment))
        self.assertEqual(session["data"], [*SESSION["data"], MASK])
        self.assertEqual(session["background"], SESSION["background"])
        self.assertEqual(session["params"], SESSION["params"])
        shaders = session["visualizations"][0]["shaders"]
        self.assertEqual(
            shaders["layer_shader_0"],
            SESSION["visualizations"][0]["shaders"]["layer_shader_0"],
        )
        self.assertEqual(
            shaders["mammaprint_dataset_mask_cancer"]["dataReferences"], [2]
        )
        self.assertFalse(shaders["mammaprint_dataset_mask_cancer"]["visible"])
        self.assertEqual(
            shaders["mammaprint_dataset_mask_cancer"]["params"]["use_channel0"], "r"
        )
        self.assertEqual(summary["patched_links"], 2)
        self.assertEqual(summary["missing_masks"], {"folding": 1})
        self.assertIn(b"Previous report additions", updated)
        self.assertEqual(self.patch(updated)[0], updated)

    def test_wrong_report_or_missing_gallery_stops_backfill(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            self.patch(self.original, provenance={**SOURCE, "split": "val"})
        with self.assertRaisesRegex(ValueError, "gallery"):
            self.patch(
                self.original.replace(HOST.encode(), b"https://other.example.test")
            )
        with self.assertRaisesRegex(ValueError, "gallery"):
            self.patch(
                self.original.replace(HOST.encode(), b"https://other.example.test", 1)
            )

    def test_cli_preview_apply_and_rollback(self):
        with (
            tempfile.TemporaryDirectory() as work,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            root = Path(work)
            html, meta, slides, manifest, preview = [
                root / name
                for name in (
                    "report.html",
                    "manifest.json",
                    "slides.parquet",
                    "layers.json",
                    "preview.html",
                )
            ]
            html.write_bytes(self.original)
            meta.write_text(json.dumps(SOURCE))
            slides.write_bytes(self.slides)
            manifest.write_text(json.dumps(self.layers))
            target = ["--report", str(html)]
            inputs = [
                "--report-manifest",
                str(meta),
                "--report-slides",
                str(slides),
                "--layer-manifest",
                str(manifest),
            ]
            self.assertEqual(backfill([*target, *inputs, "--preview", str(preview)]), 0)
            self.assertEqual(html.read_bytes(), self.original)
            stdout.truncate(0)
            stdout.seek(0)
            hashes = [
                "--expected-original-sha256",
                sha256(self.original),
                "--expected-updated-sha256",
                sha256(preview.read_bytes()),
            ]
            self.assertEqual(backfill([*target, *inputs, "--apply", *hashes]), 0)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["status"], "verified")
            self.assertEqual(html.read_bytes(), preview.read_bytes())
            self.assertEqual(
                backfill(
                    [*target, "--rollback", receipt["receipt_artifact"], "--apply"]
                ),
                0,
            )
            self.assertEqual(html.read_bytes(), self.original)

    def test_new_reports_automatically_use_preparation(self):
        with tempfile.TemporaryDirectory() as work, redirect_stdout(io.StringIO()):
            root = Path(work)
            base, meta, slides, output = [
                root / name
                for name in (
                    "base.html",
                    "manifest.json",
                    "slides.parquet",
                    "report.html",
                )
            ]
            base.write_bytes(self.original)
            meta.write_text(json.dumps(SOURCE))
            slides.write_bytes(self.slides)
            with patch(
                "ml.explain_report.prepare_layer_manifest", return_value=self.layers
            ) as prepare:
                self.assertEqual(
                    build_report(
                        [
                            "--base-report",
                            str(base),
                            "--report-manifest",
                            str(meta),
                            "--report-slides",
                            str(slides),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            prepare.assert_called_once()
            self.assertEqual(output.read_bytes(), self.patch(self.original)[0])

    def test_new_report_resolves_lineage_and_reads_real_mock_tiffs(self):
        tile_run = "6" * 32
        tile_uri = f"mlflow-artifacts:/3/{tile_run}/artifacts/tiled_test"
        mock_wsi = "project/slides/mock-case.tiff"
        config = {
            "datamodule": {"test": {"levels": {"2": {"uris": {"test": tile_uri}}}}}
        }
        config_hash = hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        source = {
            **SOURCE,
            "level": 2,
            "config_hash": config_hash,
            "explanation_experiment_id": "3",
        }
        slides_data = slide_table(mock_wsi)
        documents = {
            (RUN, "explanations/manifest.json"): json.dumps(source).encode(),
            (RUN, "explanations/slides.parquet"): slides_data,
            (RUN, "explanations/configs/config-resolved.yaml"): yaml.safe_dump(
                config
            ).encode(),
            (tile_run, "configs/config-resolved.yaml"): yaml.safe_dump(
                {
                    "data_name": "tiled_test",
                    "level": 2,
                    "dataset": {"paths": {"qc_masks": "/mnt/project/qc"}},
                    "blur_threshold": 0.5,
                    "folding_threshold": 0.1,
                }
            ).encode(),
        }
        tags = {}

        def list_artifacts(run_id, folder):
            return [
                SimpleNamespace(path=name, is_dir=False)
                for owner, name in documents
                if owner == run_id and str(Path(name).parent) == folder
            ]

        def download_artifacts(run_id, name, directory):
            output = Path(directory) / Path(name).name
            output.write_bytes(documents[run_id, name])
            return str(output)

        def log_artifact(run_id, file, folder):
            local = Path(file)
            documents[run_id, f"{folder}/{local.name}"] = local.read_bytes()

        fake = SimpleNamespace(
            list_artifacts=list_artifacts,
            download_artifacts=download_artifacts,
            log_artifact=log_artifact,
            set_tag=lambda run_id, key, value: tags.update({key: value}),
            get_run=lambda run_id: SimpleNamespace(
                data=SimpleNamespace(tags=tags),
                info=SimpleNamespace(
                    run_id=run_id,
                    experiment_id="3",
                    lifecycle_stage="active",
                    status="FINISHED",
                ),
            ),
        )
        with (
            tempfile.TemporaryDirectory() as work,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            root = Path(work)
            (root / "project/slides").mkdir(parents=True)
            (root / "project/qc").mkdir()
            pixels = np.zeros((32, 32), dtype=np.uint8)
            pixels[4:11, 6:20] = 255
            for path in (
                root / mock_wsi,
                root / "project/qc/Piqe_piqe_median_activity_mask_mock-case.tiff",
            ):
                tifffile.imwrite(
                    path,
                    pixels,
                    tile=(16, 16),
                    resolution=(20_000, 20_000),
                    resolutionunit="CENTIMETER",
                    photometric="minisblack",
                )
            base, meta, slides, output = [
                root / name
                for name in ("base.html", "manifest.json", "slides.parquet", "draft")
            ]
            # The HTML embeds the WSI path inside the encoded viewer configuration.
            base.write_bytes(
                self.original.replace(b"mock-case.mrxs", b"mock-case.tiff")
            )
            meta.write_text(json.dumps(source))
            slides.write_bytes(slides_data)
            with patch("mlflow.MlflowClient", return_value=fake):
                self.assertEqual(
                    build_report(
                        [
                            "--base-report",
                            str(base),
                            "--report-manifest",
                            str(meta),
                            "--report-slides",
                            str(slides),
                            "--output-dir",
                            str(output),
                            "--tracking-uri",
                            "https://mlflow.example.test",
                            "--run-id",
                            RUN,
                            "--mask-wsi-root",
                            str(root),
                            "--mask-source-root",
                            str(root),
                        ]
                    ),
                    0,
                )
            result = json.loads(stdout.getvalue())["dataset_masks"]
            self.assertEqual(result["patched_links"], 2)
            self.assertEqual(result["layers_by_type"], {"blur": 1})
            self.assertEqual(result["missing_masks"], {"folding": 1})
            self.assertIn(
                b"mammaprint_dataset_mask_blur", (output / "report.html").read_bytes()
            )
            layers_file = output / "dataset-mask-layers.json"
            prepared = json.loads(layers_file.read_bytes())
            saved = prepared["assets"]["mock"]["blur"]
            filename = saved["viewer_path"].split("/")[-1]
            asset = output / "dataset-mask-assets" / filename
            remote_asset = f"report/dataset-mask-assets/{filename}"
            self.assertTrue(asset.is_file())
            self.assertNotIn((RUN, remote_asset), documents)
            self.assertNotIn((RUN, "report/report.html"), documents)
            self.assertTrue(
                prepared["selections"]["blur"]["selection"].endswith("> 0.5")
            )
            replay = root / "replayed"
            self.assertEqual(
                build_report(
                    [
                        "--base-report",
                        str(output / "report.html"),
                        "--report-manifest",
                        str(output / "manifest.json"),
                        "--output-dir",
                        str(replay),
                    ]
                ),
                0,
            )
            self.assertEqual((replay / "slides.parquet").read_bytes(), slides_data)
            self.assertEqual(
                (replay / "dataset-mask-assets" / filename).read_bytes(),
                asset.read_bytes(),
            )
            asset = replay / "dataset-mask-assets" / filename
            rendered = replay / "report.html"
            apply = [
                "--base-report",
                str(rendered),
                "--report-manifest",
                str(replay / "manifest.json"),
                "--tracking-uri",
                "https://mlflow.example.test",
                "--run-id",
                RUN,
                "--experiment-id",
                "3",
                "--apply",
                "--expected-output-sha256",
                sha256(rendered.read_bytes()),
            ]
            original_tiff = asset.read_bytes()
            with patch("mlflow.MlflowClient", return_value=fake):
                asset.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "dataset mask"):
                    build_report(apply)
                self.assertNotIn((RUN, "report/report.html"), documents)
                self.assertNotIn((RUN, remote_asset), documents)
                asset.write_bytes(original_tiff)
                with self.assertRaisesRegex(
                    ValueError, "different report artifact path"
                ):
                    build_report([*apply, "--report-artifact-path", "other"])
                self.assertNotIn((RUN, "other/report.html"), documents)
                self.assertNotIn((RUN, remote_asset), documents)
                documents[RUN, "explanations/slides.parquet"] = b"changed table"
                with self.assertRaisesRegex(ValueError, "slide table differs"):
                    build_report(apply)
                self.assertNotIn((RUN, remote_asset), documents)
                documents[RUN, "explanations/slides.parquet"] = slides_data
                self.assertEqual(build_report(apply), 0)
            self.assertEqual(documents[RUN, remote_asset], original_tiff)
            self.assertEqual(
                documents[RUN, "report/report.html"], rendered.read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
