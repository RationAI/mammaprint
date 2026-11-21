"""Wait locally for one published recovery, then submit its two existing reports.

Default is an offline plan. --submit explicitly authorizes waiting and submission.
No cluster resources are allocated while waiting; VPN and this process must stay on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TRACKING = "https://mlflow-s3.rationai.cloud.trusted.e-infra.cz"
SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parents[1]


def read_json(path):
    with urllib.request.urlopen(TRACKING + path, timeout=20) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Recovery metadata exceeded 2 MiB.")
    return json.loads(data)


def published_recovery(source_id, recovery_id, fetch=read_json):
    """Return verified metadata only after this exact recovery was published."""
    recovery = fetch("/api/2.0/mlflow/runs/get?run_id=" + recovery_id)["run"]
    source = fetch("/api/2.0/mlflow/runs/get?run_id=" + source_id)["run"]
    for run, expected in [(recovery, recovery_id), (source, source_id)]:
        if (
            run["info"]["run_id"] != expected
            or str(run["info"]["experiment_id"]) != "3"
            or run["info"].get("lifecycle_stage", "active") != "active"
        ):
            raise ValueError("Run identity/experiment/lifecycle mismatch.")
    tags = {x["key"]: x["value"] for x in recovery["data"].get("tags", [])}
    if tags.get("recovery.source_run_id") != source_id:
        raise ValueError("Recovery belongs to another source.")
    if recovery["info"]["status"] in {"FAILED", "KILLED"}:
        raise ValueError(
            "Recovery failed or was killed; reports will not be submitted."
        )
    original_tags = {x["key"]: x["value"] for x in source["data"].get("tags", [])}
    if (
        recovery["info"]["status"] != "FINISHED"
        or source["info"]["status"] != "FINISHED"
        or original_tags.get("recovery.publication_status") != "completed"
        or original_tags.get("recovery.recovery_run_id") != recovery_id
    ):
        return None
    manifest = fetch(
        f"/api/2.0/mlflow-artifacts/artifacts/3/{recovery_id}"
        "/artifacts/recovery/recovery-manifest.json"
    )
    digest = manifest.get("checkpoint_sha256", "")
    training_sha = manifest.get("implementation_sha", "")
    if (
        manifest.get("status") != "completed"
        or manifest.get("source_run_id") != source_id
        or manifest.get("recovery_run_id") != recovery_id
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or not re.fullmatch(r"[a-f0-9]{40}", training_sha)
        or tags.get("recovery.implementation_sha") != training_sha
        or original_tags.get("recovery.new_checkpoint_sha256") != digest
        or manifest.get("test_evaluation", {}).get("checkpoint_sha256") != digest
        or not manifest.get("test_evaluation", {}).get("metrics")
    ):
        raise ValueError("Published recovery/checkpoint/test binding mismatch.")
    return manifest


def commands(args, manifest):
    """Use existing launchers with this verified checkpoint and training pin."""
    common = ["--context", args.context, "--namespace", args.namespace]
    return [
        [
            sys.executable,
            str(SCRIPTS / "run_attention_report.py"),
            "--source",
            args.recovery_run,
            "--checkpoint-sha256",
            manifest["checkpoint_sha256"],
            "--job-name",
            args.job_prefix + "-html",
            "--run-name",
            args.job_prefix + "-html",
            "--output-root",
            args.output_root + "/html",
            *common,
            "--submit",
        ],
        [
            sys.executable,
            str(SCRIPTS / "run_concept_explainability.py"),
            "--source",
            args.recovery_run + ":" + manifest["implementation_sha"],
            "--git-sha",
            args.git_sha,
            "--git-ref",
            "refs/heads/feat/concept-explainability",
            "--tracking-uri",
            "http://mlflow-s3.rationai-mlflow",
            "--output-root",
            args.output_root + "/concepts",
            "--job-prefix",
            args.job_prefix + "-concepts",
            "--experiment-name",
            "MammaPrint concepts",
            "--k",
            "16",
            "--gpu",
            "A40",
            "--cpu",
            "8",
            "--memory",
            "64Gi",
            "--shm",
            "16Gi",
            "--ray-cpus",
            "8",
            *common,
            "--submit",
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--recovery-run", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--job-prefix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--context", default="kubas-cluster")
    parser.add_argument("--namespace", default="rationai-jobs-ns")
    parser.add_argument("--timeout-hours", type=int, default=48)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if (
        not all(
            re.fullmatch(r"[a-f0-9]{32}", x)
            for x in [args.source_run, args.recovery_run]
        )
        or not re.fullmatch(r"[a-f0-9]{40}", args.git_sha)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,28}[a-z0-9]", args.job_prefix)
        or (args.context, args.namespace) != ("kubas-cluster", "rationai-jobs-ns")
        or not 1 <= args.timeout_hours <= 168
        or not args.state_dir.is_absolute()
        or Path(args.output_root).parts[:3] != ("/", "mnt", "projects")
        or ".." in Path(args.output_root).parts
        or len(Path(args.output_root).parts) < 4
    ):
        parser.error("Invalid source, pin, job prefix, paths or fixed cluster target.")
    print(
        json.dumps(
            {
                "mode": "wait-and-submit" if args.submit else "plan",
                "source": args.source_run,
                "recovery": args.recovery_run,
                "resources_while_waiting": "local process only; no GPU/cluster allocation",
                "reports": ["CPU tile HTML", "A40 K16 deployed concepts"],
                "timeout_hours": args.timeout_hours,
            }
        ),
        flush=True,
    )
    if not args.submit:
        return
    os.umask(0o077)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    (args.state_dir / "pid").write_text(str(os.getpid()))
    submitted = []

    def status(phase):
        (args.state_dir / "status.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "submitted": submitted,
                    "pid": os.getpid(),
                    "updated_unix": time.time(),
                    "source": args.source_run,
                    "recovery": args.recovery_run,
                },
                indent=2,
            )
        )

    try:
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            status("waiting_for_published_recovery")
            try:
                manifest = published_recovery(args.source_run, args.recovery_run)
            except (urllib.error.URLError, TimeoutError):
                manifest = None  # VPN/network interruption; retry within the bound.
            if manifest is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Recovery publication wait expired.")
            time.sleep(60)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        if head != args.git_sha or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPOSITORY
        ):
            raise ValueError("Local checkout changed; inspect before submitting.")
        requests = commands(args, manifest)
        (args.state_dir / "commands.json").write_text(json.dumps(requests, indent=2))
        (args.state_dir / "verified-recovery.json").write_text(
            json.dumps(manifest, indent=2)
        )
        for name, command in zip(["html", "concepts"], requests, strict=True):
            status("submitting_" + name)
            with (args.state_dir / (name + "-submission.log")).open("w") as output:
                subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            submitted.append(name)
        status("submitted_both")
        print("Both report jobs submitted; inspect their cluster status.", flush=True)
    except Exception:
        status("failed_inspect_before_retry")
        raise


if __name__ == "__main__":
    main()
