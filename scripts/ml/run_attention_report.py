"""Reuse the verified CPU tile-report worker; --submit creates one job.

Planning reads only the selected MLflow run/config. The frozen worker is the
same one used for the November 17 M1 report; only prepared/plan.json changes.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
import urllib.request
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[2] / "third_party/rationai-report-worker"
SERVER = "https://rancher.cloud.trusted.e-infra.cz/k8s/clusters/c-m-lr2wzbk8"
RESOURCES = {
    "cpu": 8,
    "memory": "16Gi",
    "gpu": None,
    "ephemeral_storage": "20Gi",
    "shared_scratch_budget_gib": 160,
    "shm": "1Gi",
}


def frozen_files() -> dict[str, bytes]:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    blob = (ASSETS / "worker.tar.gz").read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Frozen report worker checksum mismatch.")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            if (
                not member.isfile()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
            ):
                raise ValueError("Unsafe frozen worker member.")
            files[member.name] = archive.extractfile(member).read()
    if {k: hashlib.sha256(v).hexdigest() for k, v in files.items()} != manifest[
        "files"
    ]:
        raise ValueError("Frozen report worker file identity mismatch.")
    return files


def build_job(args, run: dict, config_bytes: bytes) -> tuple[dict, dict]:
    import yaml

    if (
        run["info"]["run_id"] != args.source
        or run["info"]["status"] != "FINISHED"
        or str(run["info"]["experiment_id"]) != "3"
    ):
        raise ValueError("Expected one finished source in MammaPrint experiment 3.")
    cfg = yaml.safe_load(config_bytes)
    graph = cfg["ml"]
    profiles = {
        ("ml.models.aggregators.AttentionMIL", "torch.nn.MSELoss", "index"): "config1",
        ("ml.models.aggregators.MeanPool", "torch.nn.MSELoss", "index"): "config2",
        (
            "ml.models.aggregators.MeanPool",
            "torch.nn.BCEWithLogitsLoss",
            "type",
        ): "config3",
    }
    profile = profiles.get(
        (graph["aggregator"]["_target_"], graph["loss"]["_target_"], cfg["label_mode"])
    )
    required = {
        "encoder": "ml.models.encoders.IdentityEncoder",
        "head": "ml.models.heads.LinearHead",
    }
    if (
        profile is None
        or cfg["feature_dim"] != 2560
        or any(graph[key]["_target_"] != target for key, target in required.items())
        or graph.get("output_activation", {}).get("_target_", "torch.nn.Identity")
        != "torch.nn.Identity"
    ):
        raise ValueError(
            "Expected a native linear attention regressor or mean regression/BCE model."
        )
    levels = cfg["dataset"]["levels"]
    if len(levels) != 1:
        raise ValueError("Expected one embedding level.")
    metrics = {
        m["key"]: m["value"]
        for m in run["data"].get("metrics", [])
        if m["key"].startswith("test/") and not m["key"].startswith("test/comparison_")
    }
    f1_key = "test/thresholded_f1" if cfg["label_mode"] == "index" else "test/f1"
    if f1_key not in metrics:
        raise ValueError("Source test evaluation is missing.")
    files = frozen_files()
    plan = json.loads(files["prepared/plan.json"])
    job = copy.deepcopy(next(job for job in plan["jobs"] if job["key"] == profile))
    job.update(
        key="selected",
        job_name=args.job_name,
        run_name=args.run_name,
        source_run_id=args.source,
        model=(
            f"L{int(next(iter(levels)))} linear + "
            f"{'attention' if profile == 'config1' else 'mean'}, "
            f"{'regression' if cfg['label_mode'] == 'index' else 'BCE'}"
        ),
        checkpoint_uri=f"mlflow-artifacts:/3/{args.source}/artifacts/checkpoints/best",
        checkpoint_sha256=args.checkpoint_sha256,
        source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        seed=cfg["seed"],
        level=int(next(iter(levels))),
        expected_test_metrics=metrics,
        resources=RESOURCES,
    )
    plan.update(
        jobs=[job],
        scratch_root=args.output_root,
        expected_slides=args.expected_slides,
        tracking_uri="http://mlflow-s3.rationai-mlflow",
        lookup_uri=args.tracking_uri,
    )
    files["prepared/plan.json"] = (json.dumps(plan, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    blob = gzip.compress(buffer.getvalue(), mtime=0)
    digest = hashlib.sha256(blob).hexdigest()
    encoded = base64.b64encode(blob).decode()
    env = [
        {"name": f"MAMMAPRINT_SOURCE_{i:03}", "value": encoded[start : start + 32000]}
        for i, start in enumerate(range(0, len(encoded), 32000))
    ]
    names = [e["name"] for e in env]
    decoder = f"import base64,hashlib,io,os,tarfile;b=base64.b64decode(''.join(os.environ[k] for k in {names!r}));assert hashlib.sha256(b).hexdigest()=={digest!r};tarfile.open(fileobj=io.BytesIO(b),mode='r:gz').extractall('.',filter='data')"
    commands = [
        "set +x",
        "set -euo pipefail",
        "umask 077",
        'git config --global --replace-all url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "ssh://git@gitlab.ics.muni.cz/"',
        'git config --global --add url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "https://gitlab.ics.muni.cz/"',
        'git config --global --add url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "git@gitlab.ics.muni.cz:"',
        "export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void",
        "export UV_NO_CACHE=1 MPLCONFIGDIR=/tmp/mammaprint-mpl XDG_CACHE_HOME=/tmp/mammaprint-cache",
        "export NO_PROXY=mlflow-s3.rationai-mlflow,mlflow.rationai-mlflow,svc.cluster.local,localhost,127.0.0.1",
        "unset MLFLOW_RUN_ID",
        "prepared_code_dir=$(mktemp -d /tmp/mammaprint-explain-code.XXXXXX)",
        'cd "$prepared_code_dir"',
        shlex.join(["python3", "-c", decoder]),
        "uv sync --frozen --no-dev",
        "uv run --no-sync -m prepared.run_job --job selected",
    ]
    template = json.loads((ASSETS / "pod-template.json").read_text())
    template["metadata"] = {"labels": {"app": args.job_name}}
    container = template["spec"]["containers"][0]
    container.update(
        name=args.job_name, command=["/bin/bash", "-c", ";".join(commands)]
    )
    container["env"].extend(env)
    body = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": args.job_name, "namespace": args.namespace},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": template,
        },
    }
    methods = ["leave_one_out", "single"]
    if job["aggregator"] == "AttentionMIL":
        methods.append("attention")
    summary = {
        "mode": "submit" if args.submit else "plan",
        "job_name": args.job_name,
        "source_run_id": args.source,
        "source_config_sha256": job["source_config_sha256"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "payload_sha256": digest,
        "context": args.context,
        "namespace": args.namespace,
        "server": SERVER,
        "resources": RESOURCES,
        "expected_slides": args.expected_slides,
        "methods": methods,
        "profile": profile,
        "split": "test",
        "worker_code_unchanged": True,
        "source_read_only": True,
    }
    return summary, body


def fetch(path: str, base: str) -> bytes:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=20) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Metadata response exceeded 2 MiB.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--tracking-uri", default="https://mlflow-s3.rationai.cloud.trusted.e-infra.cz"
    )
    parser.add_argument("--context", default="kubas-cluster")
    parser.add_argument("--namespace", default="rationai-jobs-ns")
    parser.add_argument("--expected-slides", type=int, default=308)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[a-f0-9]{32}", args.source)
        or not re.fullmatch(r"[a-f0-9]{64}", args.checkpoint_sha256)
        or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", args.job_name)
    ):
        parser.error("Invalid source, checkpoint checksum or job name.")
    if (args.context, args.namespace) != ("kubas-cluster", "rationai-jobs-ns"):
        parser.error("This workflow targets kubas-cluster/rationai-jobs-ns.")
    if (
        args.tracking_uri.rstrip("/")
        != "https://mlflow-s3.rationai.cloud.trusted.e-infra.cz"
    ):
        parser.error("Unexpected metadata endpoint.")
    output = Path(args.output_root)
    if (
        not output.is_absolute()
        or output.parts[:3] != ("/", "mnt", "projects")
        or ".." in output.parts
        or len(output.parts) < 4
    ):
        parser.error("Output root must be a fresh directory beneath /mnt/projects.")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.run_name)
        or args.expected_slides < 1
    ):
        parser.error("Invalid run name or expected slide count.")
    run = json.loads(
        fetch("/api/2.0/mlflow/runs/get?run_id=" + args.source, args.tracking_uri)
    )["run"]
    cfg = fetch(
        f"/api/2.0/mlflow-artifacts/artifacts/3/{args.source}/artifacts/configs/config-resolved.yaml",
        args.tracking_uri,
    )
    summary, body = build_job(args, run, cfg)
    print(json.dumps(summary, indent=2))
    if args.plan_file:
        args.plan_file.write_text(
            json.dumps({"summary": summary, "job": body}, indent=2)
        )
    if args.submit:
        from kubernetes import client, config

        config.load_kube_config(context=args.context)
        if client.Configuration.get_default_copy().host != SERVER:
            raise RuntimeError("Unexpected Kubernetes server.")
        api = client.BatchV1Api()
        try:
            api.read_namespaced_job(
                args.job_name, args.namespace, _request_timeout=(5, 15)
            )
        except client.rest.ApiException as error:
            if error.status != 404:
                raise
        else:
            raise RuntimeError("Job exists; inspect it before retrying.")
        api.create_namespaced_job(args.namespace, body, _request_timeout=(5, 20))
        created = api.read_namespaced_job(
            args.job_name, args.namespace, _request_timeout=(5, 15)
        )
        if not created.metadata.uid:
            raise RuntimeError("Creation not confirmed; inspect before retrying.")
        print(
            json.dumps(
                {
                    "job_name": created.metadata.name,
                    "uid": created.metadata.uid,
                    "creation_confirmed": True,
                }
            )
        )


if __name__ == "__main__":
    main()
