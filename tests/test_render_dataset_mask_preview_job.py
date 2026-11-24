from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.ml import render_dataset_mask_preview_job as renderer


RUNS = [f"{i:032x}:{i:064x}" for i in range(1, 4)]


class RenderDatasetMaskPreviewJobTest(unittest.TestCase):
    def test_preview_job_is_pinned_and_contains_only_approved_sources(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for name in renderer.SOURCES:
                source = root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"# {name}\n")
            (root / "unrelated.txt").write_text("private other data")
            output = root / "job.json"
            args = [
                "--context",
                "kubas-cluster",
                "--namespace",
                "rationai-jobs-ns",
                "--job-name",
                "mammaprint-mask-preview",
                "--experiment-id",
                "3",
                "--reuse-layer-run",
                "d" * 32,
                "--reuse-layer-batch",
                "verified-preview",
                "--output",
                str(output),
            ]
            for run in RUNS:
                args.extend(["--run", run])
            with (
                patch.object(renderer, "ROOT", root),
                redirect_stdout(io.StringIO()) as out,
            ):
                self.assertEqual(renderer.main(args), 0)
            summary = json.loads(out.getvalue())
            body = json.loads(output.read_text())
            self.assertEqual(body["metadata"]["namespace"], "rationai-jobs-ns")
            self.assertEqual(body["spec"]["backoffLimit"], 0)
            self.assertEqual(body["spec"]["parallelism"], 1)
            pod = body["spec"]["template"]["spec"]
            self.assertFalse(pod["automountServiceAccountToken"])
            container = pod["containers"][0]
            self.assertEqual(container["image"], "cerit.io/rationai/base:2.0.6")
            self.assertEqual(container["resources"]["limits"]["cpu"], "8")
            self.assertNotIn("nvidia.com/gpu", container["resources"]["limits"])
            mounts = {m["name"]: m for m in container["volumeMounts"]}
            self.assertTrue(
                all(
                    mounts[name]["readOnly"]
                    for name in ("data", "public-data", "projects")
                )
            )
            env = {item["name"]: item for item in container["env"]}
            self.assertNotIn("VCS_PASSWORD", env)
            self.assertNotIn("VCS_USER", env)
            self.assertEqual(
                env["MLFLOW_TRACKING_URI"],
                {
                    "name": "MLFLOW_TRACKING_URI",
                    "value": "http://mlflow-s3.rationai-mlflow",
                },
            )
            self.assertEqual(
                summary["tracking_uri"], env["MLFLOW_TRACKING_URI"]["value"]
            )
            self.assertIn("--stage", container["command"][-1])
            self.assertIn(
                f"--reuse-layer-run {'d' * 32} --reuse-layer-batch verified-preview",
                container["command"][-1],
            )
            self.assertEqual(summary["reuse_layer_run"], "d" * 32)
            self.assertTrue(all(run in container["command"][-1] for run in RUNS))
            packed = "".join(
                env[name]["value"]
                for name in sorted(env)
                if name.startswith(renderer.CHUNK_PREFIX)
            )
            blob = base64.b64decode(packed)
            self.assertEqual(
                hashlib.sha256(blob).hexdigest(), summary["payload_sha256"]
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                summary["manifest_sha256"],
            )
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
                self.assertEqual(set(archive.getnames()), set(renderer.SOURCES))

    def test_invalid_run_and_duplicate_run_are_rejected(self) -> None:
        for runs in (["bad"], [*RUNS, RUNS[0]]):
            with self.assertRaises((ValueError, SystemExit)):
                renderer.validate_runs(runs)

    def test_incomplete_reuse_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both"):
            renderer.render("preview", "3", RUNS, reuse_layer_run="d" * 32)


if __name__ == "__main__":
    unittest.main()
