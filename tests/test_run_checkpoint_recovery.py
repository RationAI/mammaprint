"""Offline recovery launcher contracts: no real services or training."""

import builtins
import contextlib
import importlib.util
import io
import json
import shlex
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts/ml/run_checkpoint_recovery.py"
IMPLEMENTATION = "1" * 40
TRAINING = "2" * 40
BASE = [
    "--source",
    "https://mlflow.example/#/experiments/7/runs/source-a",
    "--git-sha",
    IMPLEMENTATION,
    "--tracking-uri",
    "https://mlflow.example",
    "--output-root",
    "/mnt/projects/recovery/fresh-a",
]


def load_launcher():
    spec = importlib.util.spec_from_file_location("recovery_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load_launcher()

    def plan(self, extra=()):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.launcher.main([*BASE, *extra]), 0)
        return json.loads(output.getvalue())

    def test_import_help_and_default_are_stdlib_offline_without_mutations(self):
        original_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            if name.split(".")[0] in {
                "kube_jobs",
                "kubernetes",
                "torch",
                "mlflow",
                "hydra",
                "ml",
            }:
                raise AssertionError(f"unexpected dependency {name}")
            return original_import(name, *args, **kwargs)

        def no_io(*args, **kwargs):
            raise AssertionError("unexpected process/network/write")

        with (
            patch("builtins.__import__", side_effect=stdlib_only),
            patch.object(socket, "socket", side_effect=no_io),
            patch.object(subprocess, "Popen", side_effect=no_io),
            patch.object(Path, "mkdir", side_effect=no_io),
            patch("builtins.open", side_effect=no_io),
        ):
            self.launcher = load_launcher()
            plan = self.plan()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.launcher.main(["--help"])
            self.assertEqual(error.exception.code, 0)
        self.assertFalse(plan["remote_verified"])
        self.assertEqual(plan["mode"], "plan")
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["target"]["context"], "kubas-cluster")
        self.assertEqual(plan["resources"]["gpu"], "H100")
        self.assertEqual(plan["resources"]["cpu"], 16)
        self.assertEqual(plan["resources"]["shm"], "16Gi")
        self.assertIn("early stopping", plan["execution"]["action"])
        self.assertIn("test best validation checkpoint", plan["execution"]["action"])
        self.assertIn("comparison only", plan["execution"]["historical_score"])

    def test_generated_job_uses_pinned_implementation_and_separate_training_object(
        self,
    ):
        plan = self.plan(
            ["--training-sha", TRAINING, "--git-ref", "refs/heads/recovery"]
        )
        script = plan["jobs"][0]["script"]
        self.assertEqual(script[:2], ["set -euo pipefail", "umask 077"])
        self.assertIn("git clone --no-checkout", script[2])
        self.assertIn(f"git checkout --detach {IMPLEMENTATION}", script)
        self.assertFalse(
            any(f"checkout --detach {TRAINING}" in line for line in script)
        )
        self.assertIn(f"git cat-file -e {TRAINING}^{{commit}}", script)
        self.assertIn("uv sync --frozen", script)
        self.assertNotIn("--extra", " ".join(script))
        self.assertIn("unset MLFLOW_RUN_ID", script)
        self.assertIn("torch.cuda.is_available", " ".join(script))
        self.assertTrue(any("mkdir --" in line for line in script))
        command = shlex.split(script[-1])
        self.assertEqual(
            command[:5], ["uv", "run", "--frozen", "-m", "ml.recover_checkpoint"]
        )
        self.assertEqual(command[5], BASE[1])
        self.assertIn("--execute", command)
        self.assertIn("--implementation-sha", command)
        self.assertIn("--training-sha", command)
        self.assertNotIn("--publish", command)
        self.assertIn("--replace-original", command)
        self.assertNotIn(
            "--training-sha", shlex.split(self.plan()["jobs"][0]["script"][-1])
        )

    def test_selection_and_shell_scalars_round_trip_without_hydra(self):
        root = "/mnt/projects/[$(id)] 'quoted' ; `id`"
        job = self.plan(
            [
                "--output-root",
                root,
                "--monitor",
                "valid/thresholded_f1",
                "--mode",
                "max",
            ]
        )["jobs"][0]
        command = shlex.split(job["script"][-1])
        self.assertEqual(command[command.index("--output-root") + 1], root + "/replay")
        self.assertEqual(
            command[command.index("--monitor") + 1], "valid/thresholded_f1"
        )
        self.assertEqual(command[command.index("--mode") + 1], "max")
        mkdir = next(line for line in job["script"] if "mkdir --" in line)
        self.assertEqual(shlex.split(mkdir), ["mkdir", "--", root])

    def test_invalid_identity_pins_selection_and_resources_fail_offline(self):
        invalid = [
            ["--git-sha", "main"],
            ["--training-sha", "short"],
            ["--git-ref", "refs/../bad"],
            ["--source", "https://other.example/#/experiments/7/runs/source-a"],
            [
                "--source",
                "https://user:secret@mlflow.example/#/experiments/7/runs/source-a",
            ],
            [
                "--source",
                "https://mlflow.example/?token=secret#/experiments/7/runs/source-a",
            ],
            ["--source", "bad\nrun"],
            ["--source", "${source}"],
            ["--tracking-uri", "https://mlflow.example/?token=x"],
            ["--output-root", "/tmp/recovery"],
            ["--output-root", "/mnt/projects/a/../b"],
            ["--monitor", "valid/f1"],
            ["--mode", "max"],
            ["--monitor", "test/f1", "--mode", "max"],
            ["--cpu", "0"],
            ["--memory", "1Gi", "--shm", "2Gi"],
            ["--job-prefix", "Upper"],
            ["--username", "bad value"],
        ]
        for extra in invalid:
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.launcher.parse_args([*BASE, *extra])

    def test_submission_is_gated_and_reuses_existing_submitter(self):
        with patch.object(self.launcher.cluster, "submit_plan") as submit:
            self.plan()
            submit.assert_not_called()
            self.plan(["--submit"])
            args, plan = submit.call_args.args
            self.assertEqual(args.context, "kubas-cluster")
            self.assertEqual(args.namespace, "rationai-jobs-ns")
            self.assertEqual(plan["mode"], "submit")
            self.assertEqual(len(plan["jobs"]), 1)

    def test_mig_recovery_resources_reach_submitter_without_changing_training(self):
        resources = [
            "--gpu",
            "mig-1g.10gb",
            "--cpu",
            "8",
            "--memory",
            "32Gi",
        ]
        with patch.object(self.launcher.cluster, "submit_plan") as submit:
            plan = self.plan(resources)
            submit.assert_not_called()
            self.assertEqual(
                plan["resources"],
                {
                    "gpu": "mig-1g.10gb",
                    "gpu_limit": 1,
                    "cpu": 8,
                    "memory": "32Gi",
                    "shm": "16Gi",
                },
            )
            self.assertEqual(
                plan["jobs"][0]["script"], self.plan()["jobs"][0]["script"]
            )
            self.plan([*resources, "--submit"])
            args, submitted = submit.call_args.args
            self.assertEqual(args.gpu, "mig-1g.10gb")
            self.assertEqual(submitted["resources"], plan["resources"])

    def test_existing_submitter_rejects_wrong_target_before_loading_kubeconfig(self):
        helper = SimpleNamespace(CLUSTER="wrong", JOB_NAMESPACE="rationai-jobs-ns")
        with (
            patch.dict(sys.modules, {"kube_jobs.submit_job": helper}),
            self.assertRaisesRegex(RuntimeError, "target mismatch"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.launcher.main([*BASE, "--submit"])

    def test_actual_core_argparse_accepts_rendered_command(self):
        # The core parser handles syntax only: no metadata lookup, fit or MLflow mutation.
        from ml.recover_checkpoint import build_parser

        command = shlex.split(
            self.plan(
                ["--training-sha", TRAINING, "--monitor", "valid/f1", "--mode", "max"]
            )["jobs"][0]["script"][-1]
        )
        args = build_parser().parse_args(command[5:])
        self.assertTrue(args.execute)
        self.assertFalse(args.publish)
        self.assertTrue(args.replace_original)
        self.assertEqual(args.training_sha, TRAINING)
        self.assertEqual(args.implementation_sha, IMPLEMENTATION)

    def test_recovery_only_is_explicit_and_normal_sheet_links_are_accepted(self):
        plan = self.plan(
            ["--source", BASE[1] + "/model-metrics", "--no-replace-original"]
        )
        command = shlex.split(plan["jobs"][0]["script"][-1])
        self.assertNotIn("--replace-original", command)
        self.assertFalse(plan["execution"]["replace_original"])


if __name__ == "__main__":
    unittest.main()
