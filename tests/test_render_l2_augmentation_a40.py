import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.preprocessing import run_l2_augmentation_a40 as renderer


SOURCE_COMMIT = "a" * 40
ENCODER_REVISION = "b" * 40
OUTPUT_ROOT = "/mnt/projects/mammaprint/l2-augmentation/results"
MANIFEST = "/mnt/projects/mammaprint/l2-augmentation/selection/manifest.json"


def arguments(*extra):
    return [
        "--job-name",
        "l2-stain-pilot-1536",
        "--source-commit",
        SOURCE_COMMIT,
        "--manifest",
        MANIFEST,
        "--output-root",
        OUTPUT_ROOT,
        "--output",
        f"{OUTPUT_ROOT}/stain",
        "--view",
        "stain",
        "--seed",
        "42",
        "--encoder-revision",
        ENCODER_REVISION,
        *extra,
    ]


def render(*extra):
    with redirect_stdout(io.StringIO()) as out:
        renderer.main(arguments(*extra))
    return json.loads(out.getvalue())


def pod(job):
    return job["spec"]["template"]["spec"]


class RenderL2AugmentationA40Test(unittest.TestCase):
    def test_scoped_mounts_resources_and_pinned_pilot_argv(self):
        manifest = render(
            "--batch-size",
            "1536",
            "--workers",
            "12",
            "--pilot",
            "profile-a",
            "--pilot-slides",
            "slide1",
            "slide2",
            "--runtime-id",
            "runtime$(LITERAL)",
        )
        self.assertEqual(manifest["metadata"]["namespace"], "rationai-jobs-ns")
        self.assertEqual(
            manifest["metadata"]["annotations"]["mammaprint/cluster-context"],
            "kubas-cluster",
        )
        self.assertEqual(manifest["spec"]["backoffLimit"], 0)
        self.assertEqual(manifest["spec"]["parallelism"], 1)
        self.assertEqual(pod(manifest)["restartPolicy"], "Never")
        self.assertEqual(
            pod(manifest)["nodeSelector"], {"nvidia.com/gpu.product": "NVIDIA-A40"}
        )
        (container,) = pod(manifest)["containers"]
        self.assertEqual(
            container["resources"]["requests"], container["resources"]["limits"]
        )
        self.assertEqual(
            container["resources"]["limits"],
            {
                "cpu": "24",
                "memory": "64Gi",
                "nvidia.com/gpu": 1,
                "ephemeral-storage": "40Gi",
            },
        )
        volumes = {volume["name"]: volume for volume in pod(manifest)["volumes"]}
        self.assertEqual(
            volumes["shm"]["emptyDir"], {"medium": "Memory", "sizeLimit": "12Gi"}
        )
        self.assertEqual(volumes["scratch"]["emptyDir"], {"sizeLimit": "32Gi"})
        mounts = {mount["mountPath"]: mount for mount in container["volumeMounts"]}
        self.assertEqual(
            mounts["/mnt/projects/quality_control/resampled_mammaprint"],
            {
                "name": "projects",
                "mountPath": "/mnt/projects/quality_control/resampled_mammaprint",
                "subPath": "quality_control/resampled_mammaprint",
                "readOnly": True,
            },
        )
        self.assertTrue(
            all(
                mounts[path]["readOnly"]
                for path in (
                    "/mnt/data",
                    "/mnt/data/Public",
                    "/mnt/projects/mammaprint",
                )
            )
        )
        self.assertFalse(mounts[OUTPUT_ROOT].get("readOnly", False))
        self.assertEqual(
            mounts[OUTPUT_ROOT]["subPath"], "mammaprint/l2-augmentation/results"
        )
        env = {entry["name"]: entry for entry in container["env"]}
        self.assertEqual(
            env["HF_TOKEN"]["valueFrom"]["secretKeyRef"],
            {"name": "huggingface-secret", "key": "token"},
        )
        self.assertEqual(
            env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"], "metadata.uid"
        )
        self.assertEqual(
            env["MLFLOW_TRACKING_URI"]["value"], "http://mlflow-s3.rationai-mlflow"
        )
        script, job_name, source, telemetry, interval, ipc_bytes, *worker = container[
            "args"
        ]
        self.assertEqual(job_name, "l2-stain-pilot-1536")
        self.assertEqual(source, SOURCE_COMMIT)
        self.assertEqual(telemetry, f"{OUTPUT_ROOT}/.telemetry/{job_name}")
        self.assertEqual(interval, "5")
        self.assertGreater(int(ipc_bytes), 5 * renderer.GIB)
        self.assertLess(int(ipc_bytes), 12 * renderer.GIB)
        self.assertIn('git checkout --detach "$$source_commit"', script)
        self.assertEqual(
            worker[:7],
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-m",
                "preprocessing.l2_augmentation",
                "embed",
            ],
        )
        self.assertEqual(
            worker[-5:], ["--pilot", "profile-a", "--pilot-slides", "slide1", "slide2"]
        )
        self.assertEqual(
            worker[worker.index("--runtime-id") + 1],
            f"{renderer.IMAGE}@git:{SOURCE_COMMIT}/runtime$$(LITERAL)",
        )

    def test_rejects_undersized_or_conflicting_jobs(self):
        for extra, message in (
            (
                ("--batch-size", "1536", "--workers", "12", "--shm", "6Gi"),
                "--shm must be at least",
            ),
            (("--memory", "16Gi"), "--memory must be at least"),
            (
                ("--scratch-size", "40Gi", "--ephemeral-storage", "40Gi"),
                "scratch plus 8Gi",
            ),
            (("--output", f"{OUTPUT_ROOT}/.telemetry/output"), "reserved"),
            (("--manifest", f"{OUTPUT_ROOT}/manifest.json"), "must be disjoint"),
            (
                ("--manifest", "/mnt/projects/mammaprint/../unrelated/manifest.json"),
                "normalized absolute pod path",
            ),
            (("--pilot", "pilot"), "must be supplied together"),
            (
                (
                    "--pilot",
                    "pilot",
                    "--pilot-slides",
                    "slide1",
                    "--mlflow-run-id",
                    "c" * 32,
                ),
                "production job",
            ),
            (("--submit",), "unrecognized arguments"),
        ):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as exc:
                    renderer.main(arguments(*extra))
                self.assertEqual(exc.exception.code, 2)
                self.assertIn(message, err.getvalue())

    def test_bootstrap_records_telemetry_and_preserves_worker_exit(self):
        container = pod(render("--sample-seconds", "1"))["containers"][0]
        script, job, source, _, interval, ipc_bytes, *worker = (
            arg.replace("$$", "$") for arg in container["args"]
        )
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (root / "proc-cgroup").write_text("0::/\n")
            (cgroup / "memory.current").write_text("1234\n")
            (cgroup / "memory.peak").write_text("2345\n")
            (cgroup / "cgroup.procs").touch()
            (root / "scratch" / "mammaprint").mkdir(parents=True)
            for name, body in {
                "git": 'test "$1" = clone || test "$*" = "checkout --detach $SOURCE_COMMIT"',
                "grep": 'test "$*" = "-qx $PPID $CGROUP_PROCS"',
                "timeout": 'shift\nexec "$@"',
                "df": "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\nshm 12582912 43 12582869 1%% /dev/shm\\n'",
                "nvidia-smi": """case "$*" in
*--query-gpu=uuid,name*) printf '%s\\n' 'GPU-fixture, NVIDIA A40';;
*--id=GPU-fixture*) printf '%s\\n' 'GPU-fixture, 25, 100, 46068';;
*) exit 1;;
esac""",
                "uv": """if [ "$1" = sync ]; then
    test "$2" = --frozen && test -n "$VCS_PASSWORD"
    exit $?
fi
test "$*" = "$WORKER_ARGS" && test -z "${VCS_PASSWORD+x}" || exit 99
exit 7""",
            }.items():
                executable = bin_dir / name
                executable.write_text(f"#!/bin/sh\n{body}\n")
                executable.chmod(0o755)
            script = (
                script.replace("/sys/fs/cgroup", str(cgroup))
                .replace("/proc/self/cgroup", str(root / "proc-cgroup"))
                .replace("/scratch", str(root / "scratch"))
            )
            telemetry = root / job
            finished = subprocess.run(
                [
                    *container["command"],
                    script,
                    job,
                    source,
                    str(telemetry),
                    interval,
                    ipc_bytes,
                    *worker,
                ],
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "POD_UID": "fixture-pod",
                    "VCS_PASSWORD": "fixture-value",
                    "SOURCE_COMMIT": source,
                    "CGROUP_PROCS": str(cgroup / "cgroup.procs"),
                    "WORKER_ARGS": " ".join(worker[1:]),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(finished.returncode, 7, finished.stderr)
            self.assertNotIn("fixture-value", finished.stdout + finished.stderr)
            result = telemetry / "fixture-pod"
            self.assertEqual((result / "exit-status").read_text(), "7\n")
            self.assertIn(
                "\t1234\t2345\t44032\t12884901888\n",
                (result / "resources.tsv").read_text(),
            )
            self.assertIn(
                ",GPU-fixture, 25, 100, 46068\n", (result / "gpu.csv").read_text()
            )


if __name__ == "__main__":
    unittest.main()
