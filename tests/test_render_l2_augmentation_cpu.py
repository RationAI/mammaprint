import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.preprocessing import run_l2_augmentation_cpu as renderer


SOURCE = "a" * 40
ROOT = "/mnt/projects/mammaprint/augmentation"


def arguments(stage="prepare", *extra):
    shared = [
        stage,
        "--context",
        "kubas-cluster",
        "--namespace",
        "rationai-jobs-ns",
        "--job-name",
        f"l2-{stage}",
        "--source-commit",
        SOURCE,
        "--output-root",
        ROOT,
        "--output",
        f"{ROOT}/selection" if stage == "prepare" else f"{ROOT}/card.yaml",
    ]
    inputs = (
        [
            "--raw-tiling",
            "mlflow-artifacts:/3/run/artifacts/tiling",
            "--clean-uri",
            "runs:/clean/embeddings",
            "--data-mapping",
            "/mnt/data/mapping.csv",
            "--split",
            "train",
        ]
        if stage == "prepare"
        else [
            "--manifest",
            f"{ROOT}/selection/manifest.json",
            "--run-config",
            f"{ROOT}/stain/run.json",
            "--augmented-uri",
            f"{ROOT}/stain/embeddings",
            "--clean-val",
            "runs:/val/embeddings",
        ]
    )
    return [*shared, *inputs, *extra]


def render(stage="prepare", *extra):
    with redirect_stdout(io.StringIO()) as output:
        renderer.main(arguments(stage, *extra))
    return json.loads(output.getvalue())


def container(job):
    return job["spec"]["template"]["spec"]["containers"][0]


class RenderL2AugmentationCpuTest(unittest.TestCase):
    def test_output_root_initialization_is_explicit_and_scoped(self):
        self.assertNotIn("initContainers", render()["spec"]["template"]["spec"])
        pod = render("prepare", "--create-output-root")["spec"]["template"]["spec"]
        (init,) = pod["initContainers"]
        self.assertEqual(
            init["command"], ["mkdir", "-p", "--", ROOT, f"{ROOT}/outputs"]
        )
        self.assertEqual(init["resources"]["limits"], {"cpu": "100m", "memory": "64Mi"})
        self.assertEqual(
            init["volumeMounts"], [{"name": "projects", "mountPath": "/mnt/projects"}]
        )

    def test_prepare_scopes_storage_cpu_resources_and_credentials(self):
        job = render("prepare", "--slides", "a", "b")
        pod = job["spec"]["template"]["spec"]
        worker = container(job)
        self.assertEqual(job["metadata"]["namespace"], "rationai-jobs-ns")
        self.assertEqual(
            job["metadata"]["annotations"],
            {
                "mammaprint/source-commit": SOURCE,
                "mammaprint/cluster-context": "kubas-cluster",
            },
        )
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertTrue(pod["securityContext"]["runAsNonRoot"])
        self.assertNotIn("nodeSelector", pod)
        self.assertEqual(worker["image"], "cerit.io/rationai/base:2.0.6")
        self.assertEqual(worker["resources"]["requests"], worker["resources"]["limits"])
        self.assertEqual(
            worker["resources"]["limits"],
            {
                "cpu": "8",
                "memory": "16Gi",
                "ephemeral-storage": "40Gi",
            },
        )
        mounts = {mount["mountPath"]: mount for mount in worker["volumeMounts"]}
        for path in ("/mnt/data", "/mnt/data/Public", "/mnt/projects/mammaprint"):
            self.assertTrue(mounts[path]["readOnly"])
        self.assertEqual(mounts[ROOT]["subPath"], "mammaprint/augmentation")
        self.assertFalse(mounts[ROOT].get("readOnly", False))
        volumes = {volume["name"]: volume for volume in pod["volumes"]}
        self.assertEqual(volumes["scratch"]["emptyDir"]["sizeLimit"], "32Gi")
        self.assertTrue(volumes["data"]["persistentVolumeClaim"]["readOnly"])
        env = {entry["name"]: entry for entry in worker["env"]}
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HF_HOME", env)
        self.assertEqual(
            env["VCS_PASSWORD"]["valueFrom"]["secretKeyRef"],
            {
                "name": "gitlab-deploy-token",
                "key": "password",
            },
        )
        self.assertEqual(
            env["MLFLOW_TRACKING_URI"]["value"], "http://mlflow-s3.rationai-mlflow"
        )
        script, name, commit, *argv = worker["args"]
        self.assertTrue(script.startswith("set +x\nset -euo pipefail\n"))
        self.assertIn('exec "$$@"', script)
        self.assertEqual(name, "l2-prepare")
        self.assertEqual(commit, SOURCE)
        self.assertEqual(
            argv[:7],
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-m",
                "preprocessing.l2_augmentation",
                "prepare",
            ],
        )
        self.assertEqual(argv[-3:], ["--slides", "a", "b"])
        self.assertNotIn("--context", argv)

    def test_card_and_validation_control_forwarding_and_custom_resources(self):
        card = container(
            render(
                "card",
                "--clean-val",
                f"{ROOT}/clean-val/embeddings",
                "--clean-train",
                f"{ROOT}/clean-train/embeddings",
                "--fresh-val-manifest",
                f"{ROOT}/val/manifest.json",
                "--cpu",
                "4",
                "--memory",
                "8Gi",
                "--scratch-size",
                "512Mi",
            )
        )
        self.assertEqual(
            card["resources"]["limits"],
            {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "8704Mi"},
        )
        argv = card["args"][3:]
        self.assertEqual(argv[6], "card")
        for flag in (
            "--manifest",
            "--run-config",
            "--augmented-uri",
            "--clean-val",
            "--clean-train",
            "--fresh-val-manifest",
            "--output",
        ):
            self.assertIn(flag, argv)
        control = container(render("prepare", "--split", "val", "--full-clean-val"))
        self.assertEqual(control["args"][-1], "--full-clean-val")
        escaped = container(render("prepare", "--clean-uri", f"{ROOT}/literal$(VALUE)"))
        self.assertIn(f"{ROOT}/literal$$(VALUE)", escaped["args"])

    def test_rejects_unsafe_or_inconsistent_arguments(self):
        cases = [
            ("prepare", ["--source-commit", "short"], "40 lowercase hex"),
            ("prepare", ["--namespace", "Wrong"], "DNS label"),
            ("prepare", ["--context", ""], "intended cluster"),
            ("prepare", ["--cpu", "0"], "positive integer"),
            ("prepare", ["--memory", "0Gi"], "positive whole"),
            (
                "prepare",
                ["--scratch-size", "40Gi", "--ephemeral-storage", "40Gi"],
                "scratch plus 8Gi",
            ),
            ("prepare", ["--data-mapping", "/etc/passwd"], "Inputs must be below"),
            (
                "prepare",
                ["--clean-uri", "/mnt/projects/other/clean"],
                "Inputs must be below",
            ),
            (
                "prepare",
                ["--raw-tiling", "https://example.org/tiling"],
                "normalized absolute",
            ),
            ("prepare", ["--clean-uri", "runs:/foo/../bar"], "cannot contain"),
            ("prepare", ["--output-root", "/mnt/projects/mammaprint"], "must be below"),
            ("prepare", ["--output", ROOT], "strictly below"),
            ("prepare", ["--output", f"{ROOT}/../escape"], "normalized absolute"),
            ("prepare", ["--slides", "a", "a"], "unique"),
            ("prepare", ["--split", "val"], "requires --slides"),
            ("prepare", ["--full-clean-val"], "requires --split val"),
            (
                "prepare",
                ["--split", "val", "--slides", "a", "--full-clean-val"],
                "without --slides",
            ),
            ("card", ["--output", f"{ROOT}/card.json"], "YAML path"),
            (
                "card",
                ["--manifest", "/home/user/manifest.json"],
                "Inputs must be below",
            ),
            (
                "card",
                ["--clean-train", f"{ROOT}/clean/embeddings"],
                "supplied together",
            ),
            (
                "card",
                ["--fresh-val-manifest", f"{ROOT}/val/manifest.json"],
                "supplied together",
            ),
            (
                "card",
                [
                    "--clean-train",
                    "runs:/train/embeddings",
                    "--fresh-val-manifest",
                    f"{ROOT}/val/manifest.json",
                ],
                "local clean",
            ),
            ("prepare", ["--submit"], "unrecognized arguments"),
        ]
        for stage, extra, message in cases:
            with (
                self.subTest(stage=stage, extra=extra),
                redirect_stderr(io.StringIO()) as errors,
            ):
                with self.assertRaises(SystemExit) as result:
                    renderer.main(arguments(stage, *extra))
                self.assertEqual(result.exception.code, 2)
                self.assertIn(message, errors.getvalue())

    def test_bootstrap_bash_syntax_and_worker_exit_without_credential_leaks(self):
        worker = container(render())
        script, name, commit, *argv = [arg.replace("$$", "$") for arg in worker["args"]]
        syntax = subprocess.run(
            ["bash", "-n"], input=script, text=True, capture_output=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "scratch" / "mammaprint").mkdir(parents=True)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for command, body in {
                "git": 'test "$1" = clone || test "$*" = "checkout --detach $SOURCE_COMMIT"',
                "uv": """if [ "$1" = sync ]; then
    test "$2" = --frozen && test "$VCS_PASSWORD" = fixture-secret
    exit $?
fi
test "$*" = "$WORKER_ARGS" && test -z "${VCS_PASSWORD+x}" || exit 99
exit 7""",
            }.items():
                executable = bin_dir / command
                executable.write_text(f"#!/bin/sh\n{body}\n")
                executable.chmod(0o755)
            finished = subprocess.run(
                [
                    *worker["command"],
                    script.replace("/scratch", str(root / "scratch")),
                    name,
                    commit,
                    *argv,
                ],
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "VCS_PASSWORD": "fixture-secret",
                    "SOURCE_COMMIT": SOURCE,
                    "WORKER_ARGS": " ".join(argv[1:]),
                },
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(finished.returncode, 7, finished.stderr)
            self.assertNotIn("fixture-secret", finished.stdout + finished.stderr)


if __name__ == "__main__":
    unittest.main()
