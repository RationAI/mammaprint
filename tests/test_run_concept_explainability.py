"""Offline contracts for the concept job launcher; no live service access."""

import builtins
import contextlib
import importlib.util
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts/ml/run_concept_explainability.py"
IMPLEMENTATION = "1" * 40
TRAINING = "2" * 40
BASE = [
    "--source",
    f"source-a:{TRAINING}",
    "--git-sha",
    IMPLEMENTATION,
    "--tracking-uri",
    "http://mlflow-s3.rationai-mlflow",
    "--output-root",
    "/mnt/projects/concepts/trial-a",
]


def load_launcher():
    spec = importlib.util.spec_from_file_location("concept_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load_launcher()

    def plan(self, extra=()):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.launcher.main(BASE + list(extra)), 0)
        return json.loads(output.getvalue())

    def test_default_plan_runs_without_dependencies_process_network_or_writes(self):
        original_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            if name.split(".")[0] in {
                "kube_jobs",
                "kubernetes",
                "torch",
                "ray",
                "mlflow",
                "hydra",
            }:
                raise AssertionError(f"unexpected dependency {name}")
            return original_import(name, *args, **kwargs)

        def no_io(*args, **kwargs):
            raise AssertionError("unexpected process/network/filesystem mutation")

        with (
            patch("builtins.__import__", side_effect=stdlib_only),
            patch.object(socket, "socket", side_effect=no_io),
            patch.object(subprocess, "Popen", side_effect=no_io),
            patch.object(Path, "mkdir", side_effect=no_io),
            patch("builtins.open", side_effect=no_io),
        ):
            # Fresh import too: planning must not even import the heavy stack.
            self.launcher = load_launcher()
            plan = self.plan()
        self.assertEqual(plan["mode"], "plan")
        self.assertEqual(len(plan["jobs"]), 1)
        job = plan["jobs"][0]
        self.assertEqual(job["k"], 16)
        self.assertEqual(job["training_sha"], TRAINING)
        self.assertEqual(job["implementation_sha"], IMPLEMENTATION)
        self.assertEqual(plan["resources"]["gpu"], "H100")
        self.assertEqual(plan["resources"]["gpu_limit"], 1)
        self.assertFalse(plan["remote_verified"])

    def test_help_is_offline(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            self.launcher.main(["--help"])
        self.assertEqual(error.exception.code, 0)

    def test_rendered_job_pins_both_revisions_and_one_deployed_combined_action(self):
        job = self.plan(["--git-ref", "refs/heads/feat/concepts"])["jobs"][0]
        script = job["script"]
        self.assertEqual(script[0], "set -euo pipefail")
        self.assertEqual(script[1], "umask 077")
        self.assertIn("git clone --no-checkout", script[2])
        self.assertNotIn("--depth", " ".join(script))
        checkout = next(
            i for i, line in enumerate(script) if "checkout --detach" in line
        )
        source = next(i for i, line in enumerate(script) if TRAINING in line)
        sync = script.index("uv sync --frozen --extra explainability")
        self.assertLess(checkout, source)
        self.assertLess(source, sync)
        self.assertIn("git rev-parse HEAD", script[checkout + 1])
        self.assertIn(IMPLEMENTATION, script[checkout + 1])
        self.assertIn("unset MLFLOW_RUN_ID", script)
        command = shlex.split(script[-1])
        self.assertEqual(
            command[:9],
            [
                "uv",
                "run",
                "--frozen",
                "--extra",
                "explainability",
                "-m",
                "ml.concepts",
                "--config-name",
                "concepts_deployed",
            ],
        )
        self.assertIn("action=fit_explain", command)
        self.assertIn(f'training_sha="{TRAINING}"', command)
        self.assertIn("include_test=false", command)
        self.assertIn("device=cuda", command)
        self.assertIn("fit.n_components=16", command)
        self.assertIn("torch.cuda.is_available", " ".join(script))
        self.assertIn("openslide", " ".join(script))
        self.assertIn("pyvips", " ".join(script))

    def test_multi_source_k_outputs_are_unique_and_test_is_explicit(self):
        plan = self.plan(
            ["--source", f"source-b:{'3' * 40}", "--k", "8", "32", "--include-test"]
        )
        self.assertEqual(
            [(j["source_run_id"], j["k"]) for j in plan["jobs"]],
            [("source-a", 8), ("source-a", 32), ("source-b", 8), ("source-b", 32)],
        )
        self.assertEqual(len({j["job_name"] for j in plan["jobs"]}), 4)
        self.assertEqual(len({j["output_dir"] for j in plan["jobs"]}), 4)
        for job in plan["jobs"]:
            self.assertIn("include_test=true", shlex.split(job["script"][-1]))

    def test_shell_and_hydra_quoting_preserve_adversarial_scalars(self):
        experiment = 'Trial: [x,y] "; $(touch /tmp/injected) `id` \\ tail'
        root = "/mnt/projects/a b/[$(id)]'quote"
        uri = "https://mlflow.example/a%20b"
        plan = self.plan(
            [
                "--experiment-name",
                experiment,
                "--output-root",
                root,
                "--tracking-uri",
                uri,
            ]
        )
        script = plan["jobs"][0]["script"]
        command = shlex.split(script[-1])
        fields = dict(part.split("=", 1) for part in command[9:])
        self.assertEqual(
            fields["experiment_name"],
            '"Trial: [x,y] \\"; $(touch /tmp/injected) `id` \\ tail"',
        )
        self.assertTrue(json.loads(fields["output_dir"]).startswith(root + "/"))
        export = next(
            line for line in script if line.startswith("export MLFLOW_TRACKING_URI=")
        )
        self.assertEqual(shlex.split(export)[1], f"MLFLOW_TRACKING_URI={uri}")

    def test_rejects_invalid_inputs_before_submission(self):
        invalid = [
            ["--source", f"source-a:{TRAINING}"],
            ["--source", f"source-a:{'3' * 40}"],
            ["--source", "bad/run:" + TRAINING],
            ["--git-sha", "main"],
            ["--git-ref", "--upload-pack=evil"],
            ["--git-ref", "a\nref"],
            ["--cpu", "0"],
            ["--ray-cpus", "17"],
            ["--memory", "0Gi"],
            ["--memory", "many"],
            ["--shm", "128Gi"],
            ["--k", "0"],
            ["--k", "8", "8"],
            ["--job-prefix", "UPPER"],
            ["--job-prefix", "a" * 64],
            ["--output-root", "relative/path"],
            ["--output-root", "/mnt/projects"],
            ["--output-root", "/mnt/projects/../data"],
            ["--output-root", "/tmp/run"],
            ["--tracking-uri", "http://user:secret@mlflow"],
            ["--tracking-uri", "https://mlflow/?token=secret"],
            ["--tracking-uri", "file:///tmp/mlflow"],
            ["--tracking-uri", "https://mlflow/#secret"],
            ["--experiment-name", "null\0byte"],
            ["--experiment-name", "line\nbreak"],
            ["--experiment-name", "tab\tindent"],
            ["--experiment-name", "${oc.env:SECRET}"],
            ["--tracking-uri", "https://mlflow:bad-port"],
            ["--image", "unapproved.example/base:latest"],
            ["--username", "Invalid Username"],
        ]
        for options in invalid:
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.launcher.main(BASE + options)
            self.assertEqual(error.exception.code, 2)

    def test_names_are_valid_deterministic_and_resource_overrides_are_rendered(self):
        extra = [
            "--job-prefix",
            "a" * 30,
            "--gpu",
            "A40",
            "--cpu",
            "8",
            "--ray-cpus",
            "4",
            "--memory",
            "32Gi",
            "--shm",
            "8192Mi",
            "--k",
            "8",
            "--k",
            "32",
        ]
        first = self.plan(extra)
        second = self.plan(extra)
        self.assertEqual(first, second)
        self.assertEqual(
            first["resources"],
            {
                "gpu": "A40",
                "gpu_limit": 1,
                "cpu": 8,
                "memory": "32Gi",
                "shm": "8192Mi",
                "ray_cpus": 4,
            },
        )
        for job in first["jobs"]:
            self.assertLessEqual(len(job["job_name"]), 63)
            self.assertRegex(job["job_name"], r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
            self.assertIn("ray_cpus=4", shlex.split(job["script"][-1]))

    def execute_script_with_fake_tools(self, fail_checkout=False):
        """Run real bash control/quoting with fake external executables only."""
        experiment = 'Special "quotes"; $(false) `false` [a,b] \\ end'
        job = self.plan(["--experiment-name", experiment])["jobs"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin"
            binary.mkdir()
            log = root / "events.jsonl"
            tool = f"""#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['EVENT_LOG'], 'a') as handle:
    handle.write(json.dumps([name, args, os.getenv('MLFLOW_RUN_ID'), os.getenv('MLFLOW_TRACKING_URI')]) + '\\n')
if name == 'git':
    if args[0] == 'clone':
        pathlib.Path(args[-1]).mkdir()
    elif args[0] == 'checkout' and os.environ['FAIL_CHECKOUT'] == '1':
        sys.exit(23)
    elif args[0] == 'rev-parse':
        print('{IMPLEMENTATION}')
    elif args[0] == 'cat-file' and args[-1].startswith('{TRAINING}'):
        if not pathlib.Path('training-fetched').exists():
            sys.exit(1)
    elif args[0] == 'fetch' and args[-1] == '{TRAINING}':
        pathlib.Path('training-fetched').touch()
"""
            for name in ("git", "uv"):
                executable = binary / name
                executable.write_text(tool)
                executable.chmod(0o700)
            environment = dict(
                os.environ,
                PATH=str(binary),
                EVENT_LOG=str(log),
                MLFLOW_RUN_ID="must-be-unset",
                MLFLOW_TRACKING_URI="must-be-overridden",
                FAIL_CHECKOUT=str(int(fail_checkout)),
            )
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", ";".join(job["script"])],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            events = [json.loads(line) for line in log.read_text().splitlines()]
        return result, events

    def test_generated_shell_fetches_missing_training_and_passes_literal_arguments(
        self,
    ):
        result, events = self.execute_script_with_fake_tools()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            [
                "git",
                ["fetch", "origin", TRAINING],
                "must-be-unset",
                "must-be-overridden",
            ],
            events,
        )
        final = events[-1]
        self.assertEqual(final[0], "uv")
        self.assertEqual(final[2], None)
        self.assertEqual(final[3], "http://mlflow-s3.rationai-mlflow")
        fields = dict(value.split("=", 1) for value in final[1][8:])
        self.assertEqual(
            fields["experiment_name"],
            '"Special \\"quotes\\"; $(false) `false` [a,b] \\ end"',
        )

    @unittest.skipUnless(
        importlib.util.find_spec("hydra"),
        "actual Hydra parser requires app environment",
    )
    def test_hydra_parser_preserves_quotes_backslashes_and_punctuation(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        values = [
            "ordinary \\ slash",
            'quote " and \\"quote',
            "end \\",
            "two \\\\ end",
            "Café: [a,b] 'single'",
        ]
        for value in values:
            with self.subTest(value=value):
                plan = self.plan(["--experiment-name", value])
                overrides = shlex.split(plan["jobs"][0]["script"][-1])[9:]
                parsed = {
                    item.key_or_group: item.value()
                    for item in OverridesParser.create().parse_overrides(overrides)
                }
                self.assertEqual(parsed["experiment_name"], value)

    def test_failed_checkout_stops_before_dependency_setup_or_core_execution(self):
        result, events = self.execute_script_with_fake_tools(fail_checkout=True)
        self.assertEqual(result.returncode, 23)
        self.assertFalse(any(event[0] == "uv" for event in events))


class FakeApiError(Exception):
    def __init__(self, status):
        self.status = status


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load_launcher()
        self.created = {}
        self.calls = []
        self.helper = SimpleNamespace(
            CLUSTER="kubas-cluster", JOB_NAMESPACE="rationai-jobs-ns"
        )

        def create(**kwargs):
            self.calls.append(kwargs)
            self.created[kwargs["job_name"]] = SimpleNamespace(
                metadata=SimpleNamespace(name=kwargs["job_name"], uid="new-uid")
            )

        self.helper.submit_job = create

        def read(name, namespace):
            self.assertEqual(namespace, "rationai-jobs-ns")
            if name not in self.created:
                raise FakeApiError(404)
            return self.created[name]

        self.api = SimpleNamespace(read_namespaced_job=read)
        self.config_calls = []
        self.kubernetes = SimpleNamespace(
            config=SimpleNamespace(
                load_kube_config=lambda **kwargs: self.config_calls.append(kwargs)
            ),
            client=SimpleNamespace(
                BatchV1Api=lambda: self.api,
                Configuration=SimpleNamespace(
                    get_default_copy=lambda: SimpleNamespace(
                        host="https://cluster.example:6443"
                    )
                ),
                rest=SimpleNamespace(ApiException=FakeApiError),
            ),
        )
        self.storage = SimpleNamespace(
            secure=SimpleNamespace(DATA="data-ref", PROJECTS="projects-ref")
        )

    def run_submit(self, extra=()):
        modules = {
            "kube_jobs.submit_job": self.helper,
            "kube_jobs.storage": self.storage,
            "kubernetes": self.kubernetes,
        }
        output = io.StringIO()
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(output):
            result = self.launcher.main([*BASE, "--submit", *extra])
        return result, output.getvalue()

    def test_submit_confirms_creation_and_passes_resources_mounts(self):
        result, output = self.run_submit()
        self.assertEqual(result, 0)
        self.assertIn("new-uid", output)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertFalse(call["public"])
        self.assertEqual(call["gpu"], "H100")
        self.assertEqual(call["cpu"], 16)
        self.assertEqual(call["memory"], "64Gi")
        self.assertEqual(call["storage"], ["data-ref", "projects-ref"])
        self.assertEqual(self.config_calls, [{"context": "kubas-cluster"}])

    def test_target_mismatch_aborts_before_any_kube_loading_or_create(self):
        for target in ["CLUSTER", "JOB_NAMESPACE"]:
            with self.subTest(target=target):
                old = getattr(self.helper, target)
                setattr(self.helper, target, "unexpected")
                with self.assertRaisesRegex(RuntimeError, "target"):
                    self.run_submit()
                setattr(self.helper, target, old)
        with self.assertRaisesRegex(RuntimeError, "target"):
            self.run_submit(["--context", "other-cluster"])
        with self.assertRaisesRegex(RuntimeError, "target"):
            self.run_submit(["--namespace", "other-namespace"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.config_calls, [])

    def test_helper_swallowed_create_failure_is_not_success(self):
        self.helper.submit_job = lambda **kwargs: None
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            self.run_submit()

    def test_existing_named_job_cannot_masquerade_as_new_submission(self):
        plan = self.launcher.build_plan(self.launcher.parse_args(BASE))
        name = plan["jobs"][0]["job_name"]
        self.created[name] = SimpleNamespace(
            metadata=SimpleNamespace(name=name, uid="old")
        )
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.run_submit()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
