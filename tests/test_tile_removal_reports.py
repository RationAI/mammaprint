from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from ml.explain_report import _publish_new
from ml.explain_report import main as build_report
from ml.explainability.tile_removal_report import (
    ReportModel,
    load_report_model,
    render_blinded_review,
    render_standalone_report,
    sha256,
    update_report_html,
)
from scripts.ml.backfill_tile_removal_reports import (
    LocalReport,
    MlflowReport,
    backfill_report,
    report_slide_ids,
    rollback_report,
)
from scripts.ml.backfill_tile_removal_reports import (
    main as backfill_cli,
)


_RUN_ID = "a" * 32
_SOURCE_ID = "b" * 32
_CHECKPOINT = "c" * 64
_ARTIFACT = "report/report.html"
_TRACKING_URI = "https://mlflow.example.test"
_EXPERIMENT = "3"


def _original(title: str, cards: str) -> bytes:
    return (
        f"<!doctype html>\r\n<html><head><title>{title}</title></head><body>"
        f'<a href="/#/runs/{_RUN_ID}">Original run</a>'
        f"{cards}"
        '<script>const text = "</body>";</script>'
        "</body></html>"
    ).encode()


_RATIONAI = _original(
    "Test Report",
    "<h1>MammaPrint tile explanations</h1>"
    '<div id="metrics">Original metrics</div><a href="/xopat?test">Test card</a>',
)
_STANDALONE = _original(
    "MammaPrint tile-explainability review",
    '<article data-slide-id="val-aligned">Original validation card</article>'
    '<article data-slide-id="val-only-html">Another original card</article>',
)


class _FakeMlflow:
    def __init__(self):
        self.artifacts = {_ARTIFACT: _RATIONAI}
        self.writes = []
        self.fail_report = False
        self.after_write_error = False
        self.experiment = _EXPERIMENT
        self.tags = {"report.artifact": _ARTIFACT}

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                experiment_id=self.experiment,
                status="FINISHED",
                lifecycle_stage="active",
            ),
            data=SimpleNamespace(tags=self.tags.copy()),
        )

    def set_tag(self, run_id, key, value):
        self.tags[key] = value

    def download_artifacts(self, run_id, artifact, directory):
        self_case = Path(directory) / PurePosixPath(artifact).name
        self_case.write_bytes(self.artifacts[artifact])
        return self_case

    def log_artifact(self, run_id, file, artifact_path):
        path = Path(file)
        name = f"{artifact_path}/{path.name}"
        self.writes.append(name)
        if self.fail_report and name == _ARTIFACT and not self.after_write_error:
            raise OSError("upload failed")
        self.artifacts[name] = path.read_bytes()
        if self.fail_report and name == _ARTIFACT:
            raise OSError("write completed but response failed")

    def list_artifacts(self, run_id, prefix):
        prefix = prefix.rstrip("/") + "/"
        return [
            SimpleNamespace(path=prefix + entry)
            for entry in {
                name.removeprefix(prefix).split("/", 1)[0]
                for name in self.artifacts
                if name.startswith(prefix)
            }
        ]


class TestTileRemovalReport(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.identity = {
            "model_id": "EAN5",
            "source_run_id": _SOURCE_ID,
            "checkpoint_sha256": _CHECKPOINT,
        }
        self.model = {
            **self.identity,
            "summary": {
                "budgets": {
                    p: {"successful": 1, "evaluable": 1} for p in ("1%", "5%", "10%")
                }
            },
            "slides": [
                {
                    "slide_id": "val-aligned",
                    "status": "ok",
                    "intact_index": -0.07,
                    "intact_class": "B",
                    "target_class": "A",
                    "masks": [
                        {
                            "mask_id": "m-1",
                            "kind": "exact",
                            "masked_index": 0.02,
                            "masked_class": "A",
                            "deleted_count": 1,
                            "deleted_fraction": 0.01,
                            "verification_status": "verified",
                            "viewer_url": "https://xopat.example.test/?r=0&v=1",
                            "target_reached": True,
                        },
                        {
                            "mask_id": "m-2",
                            "kind": "compact",
                            "masked_index": -0.03,
                            "masked_class": "B",
                            "deleted_count": 2,
                            "deleted_fraction": 0.02,
                            "verification_status": "target_not_reached",
                            "viewer_url": "javascript:alert(1)",
                            "overlay_path": "overlays/unpublished.ome.tif",
                            "target_reached": False,
                        },
                    ],
                },
                {"slide_id": "val-only-results", "status": "missing_embedding"},
            ],
        }
        data_bytes = json.dumps({"models": [self.model]}).encode()
        manifest = {
            "split": "val",
            "report_data_sha256": sha256(data_bytes),
            "models": [self.identity],
            "pilot_slide_ids": [],
        }
        (self.bundle / "report_data.json").write_bytes(data_bytes)
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        self.provenance = {
            "explanation_run_id": _RUN_ID,
            "source_run_id": _SOURCE_ID,
            "checkpoint_sha256": _CHECKPOINT,
            "split": "test",
        }

    def test_test_report_preserves_original_html_and_gets_validation_list(self):
        model = load_report_model(self.bundle, self.provenance)
        after = update_report_html(_RATIONAI, model, explanation_run_id=_RUN_ID)
        self.assertIn(b"Targeted tile removal on validation", after)
        self.assertIn(b"val-aligned", after)
        self.assertIn(b"val-only-results", after)
        self.assertIn(b"Missing embedding", after)
        self.assertIn(b"0.02", after)
        self.assertIn(b"Target not reached", after)
        self.assertIn(b"https://xopat.example.test/?r=0&amp;v=1", after)
        self.assertNotIn(b"javascript:alert", after)
        before, tail = _RATIONAI.rsplit(b"</body>", 1)
        self.assertTrue(after.startswith(before))
        self.assertTrue(after.endswith(b"</body>" + tail))
        self.assertEqual(update_report_html(after, model), after)

    def test_standalone_validation_joins_by_technical_slide_id(self):
        model = load_report_model(self.bundle, {**self.provenance, "split": "val"})
        updated = update_report_html(_STANDALONE, model)
        self.assertIn(b"Original validation card", updated)
        self.assertIn(b' data-tile-removal-slide-id="val-aligned"', updated)
        self.assertIn(b' data-tile-removal-slide-id="val-only-html"', updated)
        self.assertIn(b"Validation result unavailable", updated)
        self.assertNotIn(b"val-only-results", updated)
        self.assertNotIn(b"Targeted tile removal on validation", updated)
        self.assertEqual(update_report_html(updated, model), updated)

    def test_published_validation_uses_roster_and_requires_one(self):
        model = load_report_model(self.bundle, {**self.provenance, "split": "val"})
        with self.assertRaisesRegex(ValueError, "roster"):
            update_report_html(_RATIONAI, model)
        data = pa.Table.from_pylist(
            [
                {"slide_id": "val-aligned", "status": "ok", "overlay_status": "ok"},
                {
                    "slide_id": "test-ignored",
                    "status": "ok",
                    "overlay_status": "failed",
                },
            ]
        )
        sink = io.BytesIO()
        pq.write_table(data, sink)
        updated = update_report_html(
            _RATIONAI, model, slide_ids=report_slide_ids(sink.getvalue())
        )
        self.assertIn(b' data-tile-removal-slide-id="val-aligned"', updated)
        self.assertNotIn(b"val-only-results", updated)
        self.assertNotIn(b"test-ignored", updated)

    def test_wrong_identity_tampering_and_bad_anchors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique match"):
            load_report_model(
                self.bundle, {**self.provenance, "checkpoint_sha256": "0" * 64}
            )
        with self.assertRaisesRegex(ValueError, "does not identify"):
            load_report_model(self.bundle, self.provenance, explanation_run_id="0" * 32)
        model = load_report_model(self.bundle, self.provenance)
        for bad in (
            _RATIONAI.replace(b"</body></html>", b"</html>"),
            _RATIONAI.replace(b"MammaPrint tile explanations", b"Another project"),
            _RATIONAI + b"</body>",
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "anchor"):
                update_report_html(bad, model)
        marker = b"<!-- BEGIN MAMMAPRINT TARGETED TILE REMOVAL -->"
        with self.assertRaisesRegex(ValueError, "markers"):
            update_report_html(_RATIONAI + marker, model)
        (self.bundle / "report_data.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_report_model(self.bundle, self.provenance)

    def test_new_standalone_and_blinded_html_use_only_allowed_fields(self):
        standalone = render_standalone_report(self.model)
        self.assertIn("Targeted tile removal", standalone)
        self.assertIn("val-only-results", standalone)
        blind = render_blinded_review(
            [
                {
                    "review_id": "uX-19",
                    "viewer_url": "https://xopat.example.test/?r=uX-19",
                    "score": -0.237,
                    "mask_type": "private_exact",
                    "slide_id": "private_slide",
                    "target_class": "private_target",
                }
            ]
        )
        self.assertIn("uX-19", blind)
        for private in ("-0.237", "private_exact", "private_slide", "private_target"):
            self.assertNotIn(private, blind)
        self.assertIn("Download responses", blind)
        with self.assertRaisesRegex(ValueError, "opaque"):
            render_blinded_review([{"review_id": '<img src="x">'}])

    def test_local_default_preview_apply_retry_and_rollback(self):
        path = self.root / "Prior report.html"
        path.write_bytes(_RATIONAI)
        store = LocalReport(path)
        preview = self.root / "preview.html"
        planned = backfill_report(store, self.bundle, self.provenance, preview=preview)
        self.assertEqual(planned["status"], "preview")
        self.assertEqual(planned["missing_overlay_references"], 1)
        self.assertEqual(path.read_bytes(), _RATIONAI)
        with self.assertRaises(FileExistsError):
            backfill_report(store, self.bundle, self.provenance, preview=path)
        applied = backfill_report(store, self.bundle, self.provenance, apply=True)
        self.assertEqual(applied["status"], "verified")
        self.assertEqual(store.read(applied["backup_artifact"]), _RATIONAI)
        self.assertEqual(path.read_bytes(), preview.read_bytes())
        second = backfill_report(store, self.bundle, self.provenance, apply=True)
        self.assertEqual(second["status"], "already_updated")
        journal = applied["receipt_artifact"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            rollback_report(store, journal, apply=True)
        path.write_bytes(preview.read_bytes())
        self.assertEqual(rollback_report(store, journal)["status"], "rollback_preview")
        self.assertEqual(
            rollback_report(store, journal, apply=True)["status"], "rolled_back"
        )
        self.assertEqual(path.read_bytes(), _RATIONAI)

    def test_new_report_and_backfill_have_same_rendered_section(self):
        original = self.root / "report.html"
        original.write_bytes(_RATIONAI)
        manifest = self.root / "original-manifest.json"
        manifest.write_text(json.dumps(self.provenance))
        new = self.root / "new.html"
        options = ["--report-manifest", str(manifest)]
        findings = self.root / "findings.txt"
        findings.write_text("Reviewed <specific> features.\nSecond observation.")
        findings_args = ["--review-findings", str(findings)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                build_report(
                    [
                        "--base-report",
                        str(original),
                        *options,
                        "--tile-removal-bundle",
                        str(self.bundle),
                        "--output",
                        str(new),
                        *findings_args,
                    ]
                ),
                0,
            )
            self.assertEqual(
                backfill_cli(
                    [
                        "--report",
                        str(original),
                        *options,
                        "--bundle",
                        str(self.bundle),
                        "--apply",
                        *findings_args,
                    ]
                ),
                0,
            )
        self.assertEqual(original.read_bytes(), new.read_bytes())
        self.assertIn(b"Reviewed &lt;specific&gt;", new.read_bytes())

    def test_pinned_worker_no_upload_runs_publisher_and_optional_composition(self):
        worker = self.root / "worker"
        package = worker / "ml"
        package.mkdir(parents=True)
        program = (
            "import sys\nfrom pathlib import Path\n"
            "assert '--no-upload' in sys.argv\n"
            "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
            f"(out / 'report.html').write_bytes({_RATIONAI!r})\n"
            f"(out / 'manifest.json').write_text({json.dumps(self.provenance)!r})\n"
            "(out / 'slides.parquet').write_bytes(b'test-only')\n"
        )
        (package / "explain_report.py").write_text(program)
        arguments = [
            "--worker-root",
            str(worker),
            "--worker-python",
            sys.executable,
            "--run-id",
            _RUN_ID,
            "--tracking-uri",
            _TRACKING_URI,
            "--no-upload",
        ]
        with (
            patch("ml.explain_report._FROZEN_ENTRYPOINT", sha256(program.encode())),
            redirect_stdout(io.StringIO()),
        ):
            ordinary = self.root / "ordinary"
            self.assertEqual(
                build_report([*arguments, "--output-dir", str(ordinary)]), 0
            )
            self.assertEqual((ordinary / "report.html").read_bytes(), _RATIONAI)
            self.assertEqual(
                json.loads((ordinary / "manifest.json").read_text()), self.provenance
            )
            enriched = self.root / "enriched.html"
            self.assertEqual(
                build_report(
                    [
                        *arguments,
                        "--output",
                        str(enriched),
                        "--tile-removal-bundle",
                        str(self.bundle),
                    ]
                ),
                0,
            )
            self.assertIn(b"Targeted tile removal on validation", enriched.read_bytes())

    def test_new_publication_verifies_and_never_replaces_existing_html(self):
        fake = _FakeMlflow()
        fake.tags.clear()
        fake.artifacts.clear()
        fake.artifacts["explanations/manifest.json"] = json.dumps(
            self.provenance
        ).encode()
        args = SimpleNamespace(
            run_id=_RUN_ID,
            experiment_id=_EXPERIMENT,
            artifact_mount="secure_mlflow",
            report_artifact_path="report",
            artifact_path="explanations",
            expected_output_sha256=sha256(_RATIONAI),
            xopat_ui_uri="https://xopat.example.test/",
            methods=("leave_one_out",),
        )
        url = _publish_new(fake, args, _RATIONAI, self.provenance)
        self.assertIn("view=report.html", url)
        self.assertEqual(fake.tags["report.artifact"], _ARTIFACT)
        self.assertEqual(fake.artifacts[_ARTIFACT], _RATIONAI)
        _publish_new(fake, args, _RATIONAI, self.provenance)
        self.assertEqual(fake.writes, [_ARTIFACT])
        changed = _RATIONAI + b"\n"
        args.expected_output_sha256 = sha256(changed)
        with self.assertRaisesRegex(ValueError, "backfill"):
            _publish_new(fake, args, changed, self.provenance)
        self.assertEqual(fake.artifacts[_ARTIFACT], _RATIONAI)

    def test_mlflow_preview_and_failed_upload_are_recoverable(self):
        fake = _FakeMlflow()
        store = MlflowReport(fake, _TRACKING_URI, _EXPERIMENT, _RUN_ID, _ARTIFACT)
        arguments = {
            "store": store,
            "bundle": self.bundle,
            "provenance": self.provenance,
            "explanation_run_id": _RUN_ID,
        }
        plan = backfill_report(**arguments)
        self.assertEqual(fake.writes, [])
        with self.assertRaisesRegex(ValueError, "approved preview"):
            backfill_report(**arguments, apply=True, expected_updated="0" * 64)
        self.assertEqual(fake.writes, [])
        fake.fail_report = True
        with self.assertRaisesRegex(RuntimeError, r"receipt .*receipt.json"):
            backfill_report(
                **arguments,
                apply=True,
                expected_original=plan["original_sha256"],
                expected_updated=plan["updated_sha256"],
            )
        self.assertEqual(fake.artifacts[_ARTIFACT], _RATIONAI)
        fake.after_write_error = True
        with self.assertRaisesRegex(RuntimeError, "roll back"):
            backfill_report(**arguments, apply=True)
        journal = next(
            name for name in reversed(fake.writes) if name.endswith("/receipt.json")
        )
        self.assertEqual(json.loads(fake.artifacts[journal])["status"], "prepared")
        fake.fail_report = False
        retry = backfill_report(**arguments, apply=True)
        self.assertEqual(retry["status"], "already_updated")
        self.assertEqual(retry["receipt_artifact"], journal)
        self.assertEqual(retry["original_sha256"], plan["original_sha256"])
        self.assertEqual(json.loads(fake.artifacts[journal])["status"], "verified")
        self.assertEqual(
            rollback_report(store, journal, apply=True)["status"], "rolled_back"
        )
        self.assertEqual(fake.artifacts[_ARTIFACT], _RATIONAI)
        with self.assertRaisesRegex(ValueError, "does not publish"):
            MlflowReport(fake, _TRACKING_URI, "999", _RUN_ID, _ARTIFACT)

    def test_refresh_replaces_generated_section_without_extra_markers(self):
        model = load_report_model(self.bundle, self.provenance)
        report = update_report_html(_RATIONAI, model)
        changed = ReportModel(
            {**self.model, "review_findings": "<b>artifact</b>"},
            model.split,
            "f" * 64,
        )
        refreshed = update_report_html(report, changed)
        self.assertEqual(refreshed.count(b"BEGIN MAMMAPRINT TARGETED"), 1)
        self.assertIn(b"&lt;b&gt;artifact&lt;/b&gt;", refreshed)
        self.assertEqual(update_report_html(refreshed, changed), refreshed)
