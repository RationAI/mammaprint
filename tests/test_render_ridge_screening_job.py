from __future__ import annotations

import base64
import gzip
import json
import os
import subprocess

import pytest

from scripts.ml import render_ridge_screening_job as renderer
from scripts.ml.render_ridge_screening_job import main


IMAGE = f"cerit.io/rationai/mammaprint@sha256:{'a' * 64}"
COMMIT = "b" * 40
BASE_COMMIT = "a" * 40


def _arguments(operation="cache", image=IMAGE):
    return [
        operation,
        "--context",
        "example-cluster",
        "--namespace",
        "example-namespace",
        "--job-name",
        "ridge-pilot",
        "--image",
        image,
        "--config",
        "/mnt/projects/ridge plans/screening.yaml",
        "--output",
        "/ridge-output/cache-pilot",
        "--output-subpath",
        "mammaprint/screening",
    ]


def test_render_cpu_pilot_uses_scoped_durable_output(capsys):
    main([*_arguments(), "--cpu", "3", "--pilot-slides", "7", "--dry-run"])

    job = json.loads(capsys.readouterr().out)
    assert job["metadata"] == {
        "name": "ridge-pilot",
        "namespace": "example-namespace",
        "annotations": {"mammaprint.rationai.org/target-context": "example-cluster"},
    }
    pod = job["spec"]["template"]["spec"]
    worker = pod["containers"][0]
    assert worker["image"] == IMAGE
    assert worker["command"] == [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "scripts.ml.run_ridge_screening",
        "cache",
        "--config",
        "/mnt/projects/ridge plans/screening.yaml",
        "--output",
        "/ridge-output/cache-pilot",
        "--pilot-slides",
        "7",
        "--dry-run",
    ]
    assert worker["resources"]["requests"] == {"cpu": "3", "memory": "16Gi"}
    assert worker["resources"]["limits"] == worker["resources"]["requests"]
    assert worker["volumeMounts"] == [
        {"name": "input", "mountPath": "/mnt/projects", "readOnly": True},
        {
            "name": "results",
            "mountPath": "/ridge-output",
            "subPath": "mammaprint/screening",
        },
        {"name": "scratch", "mountPath": "/scratch"},
    ]
    assert pod["volumes"][2] == {"name": "scratch", "emptyDir": {"sizeLimit": "64Gi"}}


def test_checkout_passes_the_pinned_sha_and_screen_arguments(tmp_path, capsys):
    main(
        [
            *_arguments("screen", "cerit.io/rationai/base:2.0.6"),
            "--source-commit",
            COMMIT,
            "--screen",
            "levels",
            "--gitlab-secret",
            "gitlab-token",
        ]
    )
    job = json.loads(capsys.readouterr().out)
    worker = job["spec"]["template"]["spec"]["containers"][0]
    env = {variable["name"]: variable for variable in worker["env"]}
    assert env["VCS_PASSWORD"] == {
        "name": "VCS_PASSWORD",
        "valueFrom": {"secretKeyRef": {"name": "gitlab-token", "key": "password"}},
    }

    git = tmp_path / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = clone ]; then mkdir -p "$TMPDIR/repo"; '
        'else printf "%s\\n" "$*"; fi\n'
    )
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "<%s>\\n" "$@"\n')
    for path in (git, uv):
        path.chmod(0o700)

    result = subprocess.run(
        worker["command"] + worker["args"],
        cwd=tmp_path,
        env={"PATH": f"{tmp_path}{os.pathsep}/bin:/usr/bin", "TMPDIR": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"checkout --detach {COMMIT}" in result.stdout
    assert "</mnt/projects/ridge plans/screening.yaml>" in result.stdout
    assert "<screen>\n" in result.stdout
    assert "<--screen>\n<levels>\n" in result.stdout


def test_cache_pilot_with_committed_overlay(tmp_path, capsys, monkeypatch):
    patch = b"fixture committed patch\n"
    encoded = base64.b64encode(gzip.compress(patch, mtime=0)).decode("ascii")
    monkeypatch.setattr(renderer, "_committed_overlay", lambda *_: encoded)
    args = _arguments(image="cerit.io/rationai/base:2.0.6")
    at = args.index("--config")
    args[at : at + 2] = ["--repo-config"]
    main(
        [
            *args,
            "--source-commit",
            COMMIT,
            "--overlay-base",
            BASE_COMMIT,
            "--pilot-slides",
            "10",
        ]
    )
    job = json.loads(capsys.readouterr().out)
    annotations = job["metadata"]["annotations"]
    assert annotations["mammaprint.rationai.org/source-commit"] == COMMIT
    assert annotations["mammaprint.rationai.org/overlay-base"] == BASE_COMMIT
    worker = job["spec"]["template"]["spec"]["containers"][0]
    git = tmp_path / "git"
    git.write_text(
        '#!/bin/sh\ncase "$1" in\n'
        'clone) mkdir -p "$TMPDIR/repo";;\n'
        "apply) cat;;\n"
        '*) printf "%s\\n" "$*";;\n'
        "esac\n"
    )
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "<%s>\\n" "$@"\n')
    for path in (git, uv):
        path.chmod(0o700)

    output = subprocess.run(
        worker["command"] + worker["args"],
        cwd=tmp_path,
        env={"PATH": f"{tmp_path}{os.pathsep}/bin:/usr/bin", "TMPDIR": str(tmp_path)},
        check=True,
        capture_output=True,
    ).stdout
    assert f"checkout --detach {BASE_COMMIT}".encode() in output
    assert patch in output
    assert b"<configs/sweeps/ridge_screening/screening.yaml>" in output
    assert b"<--pilot-slides>\n<10>" in output


@pytest.mark.parametrize(
    ("operation", "override", "error"),
    [
        ("cache", ["--image", "cerit.io/rationai/base:latest"], "image digest"),
        ("cache", ["--scratch", "/ridge-output/scratch"], "disjoint paths"),
        ("cache", ["--output", "/scratch/results"], "under /ridge-output"),
        ("cache", ["--output-subpath", "../outside"], "relative PVC directory"),
        ("cache", ["--config", "/other/config.yml"], "under --input-mount"),
        ("cache", ["--tracking-uri", "https://user:token@host"], "Secrets"),
        ("cache", ["--overlay-base", BASE_COMMIT], "--source-commit"),
        ("screen", ["--pilot-slides", "5"], "cache operation"),
    ],
)
def test_rejects_mismatched_target_inputs(operation, override, error, capsys):
    with pytest.raises(SystemExit) as exc:
        main(_arguments(operation) + override)
    assert exc.value.code == 2
    assert error in capsys.readouterr().err
