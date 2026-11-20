"""Frozen report replay changes bindings only and performs no training."""

import base64
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts/ml/run_attention_report.py"


def launcher():
    spec = importlib.util.spec_from_file_location("attention_report_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    args = SimpleNamespace(
        source="a" * 32,
        checkpoint_sha256="b" * 64,
        job_name="mammaprint-attention-test",
        run_name="attention-test",
        output_root="/mnt/projects/mammaprint/report-test",
        expected_slides=308,
        tracking_uri="https://mlflow-s3.rationai.cloud.trusted.e-infra.cz",
        context="kubas-cluster",
        namespace="rationai-jobs-ns",
        submit=False,
    )
    run = {
        "info": {"run_id": args.source, "status": "FINISHED", "experiment_id": "3"},
        "data": {"metrics": [{"key": "test/thresholded_f1", "value": 0.81268}]},
    }
    cfg = {
        "seed": 2,
        "label_mode": "index",
        "feature_dim": 2560,
        "dataset": {"levels": {"2": {}}},
        "ml": {
            key: {"_target_": target}
            for key, target in {
                "encoder": "ml.models.encoders.IdentityEncoder",
                "aggregator": "ml.models.aggregators.AttentionMIL",
                "head": "ml.models.heads.LinearHead",
                "loss": "torch.nn.MSELoss",
            }.items()
        },
    }
    return args, run, yaml.safe_dump(cfg).encode()


def test_same_frozen_worker_with_only_one_selected_plan_and_no_gpu():
    module = launcher()
    args, run, config = inputs()
    before = copy.deepcopy(run)
    summary, job = module.build_job(args, run, config)
    assert summary["worker_code_unchanged"] and summary["source_read_only"]
    assert run == before
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"] == {
        "cpu": "8",
        "memory": "16Gi",
        "ephemeral-storage": "20Gi",
    }
    assert pod["volumes"][-1]["emptyDir"]["sizeLimit"] == "1Gi"
    chunks = {
        e["name"]: e["value"]
        for e in container["env"]
        if e["name"].startswith("MAMMAPRINT_SOURCE_")
    }
    payload = base64.b64decode("".join(chunks[k] for k in sorted(chunks)))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
    original = module.frozen_files()
    assert {k for k in files if files[k] != original[k]} == {"prepared/plan.json"}
    plan = json.loads(files["prepared/plan.json"])
    assert len(plan["jobs"]) == 1
    selected = plan["jobs"][0]
    assert selected["source_run_id"] == args.source and selected["seed"] == 2
    assert selected["checkpoint_sha256"] == args.checkpoint_sha256
    assert selected["key"] == "selected"
    assert "explain.attention.enabled=true" in selected["explain_overrides"]
    assert "explain.ig.enabled=false" in selected["explain_overrides"]
    assert (
        summary["payload_sha256"]
        == module.build_job(args, run, config)[0]["payload_sha256"]
    )
    subprocess.run(
        ["bash", "-n"], input=container["command"][-1], text=True, check=True
    )
    assert container["command"][-1].startswith("set +x;")
    assert "unset MLFLOW_RUN_ID" in container["command"][-1]


@pytest.mark.parametrize("change", ["status", "identity", "graph"])
def test_rejects_wrong_source_or_graph(change):
    args, run, config = inputs()
    if change == "status":
        run["info"]["status"] = "RUNNING"
    if change == "identity":
        run["info"]["run_id"] = "c" * 32
    if change == "graph":
        config = config.replace(b"AttentionMIL", b"MeanPool")
    with pytest.raises(ValueError):
        launcher().build_job(args, run, config)


def test_snapshot_integrity_is_enforced(tmp_path, monkeypatch):
    module = launcher()
    (tmp_path / "worker.tar.gz").write_bytes(b"altered")
    (tmp_path / "manifest.json").write_bytes(
        (module.ASSETS / "manifest.json").read_bytes()
    )
    monkeypatch.setattr(module, "ASSETS", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        module.frozen_files()


def test_help_is_stdlib_only_and_never_contacts_services():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--submit" in result.stdout
