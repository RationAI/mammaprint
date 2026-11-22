"""Deferred reports must wait for publication of the exact tested checkpoint."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/ml/run_reports_after_recovery.py"


def module():
    spec = importlib.util.spec_from_file_location("deferred_reports", SCRIPT)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def evidence():
    source, recovery, digest, sha = "a" * 32, "b" * 32, "c" * 64, "d" * 40
    original = {
        "info": {"run_id": source, "experiment_id": "3", "status": "FINISHED"},
        "data": {
            "tags": [
                {"key": k, "value": v}
                for k, v in {
                    "recovery.publication_status": "completed",
                    "recovery.recovery_run_id": recovery,
                    "recovery.new_checkpoint_sha256": digest,
                }.items()
            ]
        },
    }
    run = {
        "info": {"run_id": recovery, "experiment_id": "3", "status": "FINISHED"},
        "data": {
            "tags": [
                {"key": k, "value": v}
                for k, v in {
                    "recovery.source_run_id": source,
                    "recovery.implementation_sha": sha,
                }.items()
            ]
        },
    }
    manifest = {
        "status": "completed",
        "source_run_id": source,
        "recovery_run_id": recovery,
        "checkpoint_sha256": digest,
        "implementation_sha": sha,
        "test_evaluation": {"checkpoint_sha256": digest, "metrics": {"test/f1": 0.765}},
    }
    docs = {
        "/api/2.0/mlflow/runs/get?run_id=" + source: {"run": original},
        "/api/2.0/mlflow/runs/get?run_id=" + recovery: {"run": run},
        f"/api/2.0/mlflow-artifacts/artifacts/3/{recovery}/artifacts/recovery/recovery-manifest.json": manifest,
    }
    return source, recovery, docs, original, run, manifest


def test_requires_publication_before_requesting_artifacts():
    source, recovery, docs, original, run, manifest = evidence()
    api = module()
    assert api.published_recovery(source, recovery, docs.__getitem__) == manifest
    run["info"]["status"] = "RUNNING"
    assert api.published_recovery(source, recovery, docs.__getitem__) is None
    run["info"]["status"] = "FINISHED"
    original["data"]["tags"] = []
    assert api.published_recovery(source, recovery, docs.__getitem__) is None


@pytest.mark.parametrize(
    "bad", ["source", "failed", "experiment", "checkpoint", "test", "training"]
)
def test_rejects_wrong_run_or_checkpoint_bindings(bad):
    source, recovery, docs, _original, run, manifest = evidence()
    if bad == "source":
        run["data"]["tags"][0]["value"] = "f" * 32
    if bad == "failed":
        run["info"]["status"] = "FAILED"
    if bad == "experiment":
        run["info"]["experiment_id"] = "84"
    if bad == "checkpoint":
        manifest["checkpoint_sha256"] = "f" * 64
    if bad == "test":
        manifest["test_evaluation"]["checkpoint_sha256"] = "f" * 64
    if bad == "training":
        manifest["implementation_sha"] = "f" * 40
    with pytest.raises(ValueError):
        module().published_recovery(source, recovery, docs.__getitem__)


def test_commands_keep_training_pin_separate_and_test_out_of_concepts():
    _source, recovery, _docs, _original, _run, manifest = evidence()
    args = SimpleNamespace(
        recovery_run=recovery,
        context="kubas-cluster",
        namespace="rationai-jobs-ns",
        job_prefix="selected-bce",
        output_root="/mnt/projects/selected-bce",
        git_sha="e" * 40,
    )
    html, concepts = module().commands(args, manifest)
    assert html[html.index("--source") + 1] == recovery
    assert html[html.index("--checkpoint-sha256") + 1] == manifest["checkpoint_sha256"]
    assert concepts[concepts.index("--source") + 1] == recovery + ":" + "d" * 40
    assert concepts[concepts.index("--git-sha") + 1] == "e" * 40
    assert "--include-test" not in concepts
    assert concepts[concepts.index("--memory") + 1] == "64Gi"
    assert concepts[concepts.index("--gpu") + 1] == "A40"


def test_default_is_offline_and_creates_no_waiter_or_job(tmp_path):
    state = tmp_path / "state"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--source-run",
            "a" * 32,
            "--recovery-run",
            "b" * 32,
            "--git-sha",
            "c" * 40,
            "--job-prefix",
            "selected-bce",
            "--output-root",
            "/mnt/projects/selected-bce",
            "--state-dir",
            str(state),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "plan"
    assert not state.exists()


@pytest.mark.parametrize("prefix", ["mammaprint-best-bce-s2", "b" * 30])
def test_generated_concept_command_passes_real_launcher_validation(prefix):
    _source, recovery, _docs, _original, _run, manifest = evidence()
    args = SimpleNamespace(
        recovery_run=recovery,
        context="kubas-cluster",
        namespace="rationai-jobs-ns",
        job_prefix=prefix,
        output_root="/mnt/projects/selected-bce",
        git_sha="e" * 40,
    )
    _html, concepts = module().commands(args, manifest)
    assert concepts[-1] == "--submit"
    result = subprocess.run(concepts[:-1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["jobs"][0]["job_name"].startswith(prefix + "-")
