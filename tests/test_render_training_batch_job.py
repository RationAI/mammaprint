import json

import pytest

from scripts.ml import render_training_batch_job as renderer


def arguments(*extra):
    return [
        "--context",
        "kubas-cluster",
        "--namespace",
        "rationai-jobs-ns",
        "--job-name",
        "mp-abmil-mig10-a1",
        "--batch-id",
        "abmil-pilot",
        "--source-commit",
        "b" * 40,
        "--image",
        "cerit.io/rationai/base:2.0.6",
        "--gpu",
        "mig-1g.10gb",
        "--cpu",
        "16",
        "--memory",
        "48Gi",
        "--shm",
        "8Gi",
        "--output-root",
        "/mnt/projects/mammaprint/training-batches",
        "--experiment",
        "ml/train_mil_embeddings_abmil_mean_regression",
        "--seeds",
        "101",
        "102",
        "--override",
        "test_after_fit=false",
        *extra,
    ]


def unescape(value):
    return value.replace("$$", "$")


def test_renders_one_mig_device_and_literal_seed_commands(capsys):
    literal = "metadata.description='val; $(EXAMPLE) $literal'"
    args = arguments("--max-parallel", "2", "--override", literal)
    renderer.main(args)
    rendered = capsys.readouterr().out
    renderer.main(args)
    assert capsys.readouterr().out == rendered
    job = json.loads(rendered)
    assert job["kind"] == "Job"
    assert job["metadata"]["namespace"] == "rationai-jobs-ns"
    assert (
        job["metadata"]["annotations"]["mammaprint.rationai.org/target-context"]
        == "kubas-cluster"
    )
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert len(pod["containers"]) == 1
    container = pod["containers"][0]
    assert container["resources"]["limits"] == {
        "cpu": "16",
        "memory": "48Gi",
        "nvidia.com/mig-1g.10gb": 1,
    }
    assert not pod.get("nodeSelector")
    assert any(
        v.get("persistentVolumeClaim", {}).get("claimName") == "projects-rw"
        for v in pod["volumes"]
    )
    assert any(v.get("emptyDir", {}).get("sizeLimit") == "8Gi" for v in pod["volumes"])
    assert any(
        e.get("valueFrom", {}).get("fieldRef", {}).get("fieldPath") == "metadata.uid"
        for e in container["env"]
    )
    bootstrap, _, commit, spec_json, output_root, *_ = map(unescape, container["args"])
    assert 'git -C "$WORK" checkout --detach "$1"' in bootstrap
    assert "exec .venv/bin/python -m scripts.ml.run_training_batch" in bootstrap
    assert commit == "b" * 40
    assert output_root.startswith("/mnt/projects/")
    spec = json.loads(spec_json)
    assert spec["max_parallel"] == 2
    for seed, run in zip((101, 102), spec["runs"], strict=True):
        assert run["id"] == f"seed-{seed}"
        cmd = run["argv"]
        assert cmd[:5] == ["uv", "run", "--no-sync", "python", "-m"]
        assert {
            f"seed={seed}",
            "test_after_fit=false",
            literal,
            "trainer.devices=1",
            "hydra.run.dir=${oc.env:TRAIN_RUN_DIR}",
            "hydra.job.chdir=true",
        } <= set(cmd)


def test_renders_full_gpu_resource(capsys):
    args = arguments()
    args[args.index("mig-1g.10gb")] = "A40"
    renderer.main(args)
    pod = json.loads(capsys.readouterr().out)["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-A40"}
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert json.loads(unescape(container["args"][3]))["max_parallel"] == 1


def test_reserves_local_storage_independently_of_the_scratch_volume(capsys):
    renderer.main(arguments("--scratch-size", "320Gi", "--ephemeral-storage", "336Gi"))
    pod = json.loads(capsys.readouterr().out)["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    resources = container["resources"]
    assert resources["requests"]["ephemeral-storage"] == "336Gi"
    assert resources["limits"]["ephemeral-storage"] == "336Gi"
    scratch = next(volume for volume in pod["volumes"] if volume["name"] == "scratch")
    assert scratch["emptyDir"]["sizeLimit"] == "320Gi"
    with pytest.raises(SystemExit) as result:
        renderer.main(
            arguments("--scratch-size", "320Gi", "--ephemeral-storage", "256Gi")
        )
    assert result.value.code == 2


@pytest.mark.parametrize(
    "override",
    [
        "seed=33",
        "++trainer.devices=2",
        "~logger.tags",
        "metadata={run_name:bad}",
        "hydra/job_logging=disabled",
        "--multirun",
        "trainer=abmil",
        "checkpoint=relative.ckpt",
        "checkpoint=runs:/legacy/path",
    ],
)
def test_rejects_overrides_that_conflict_with_runtime(override):
    with pytest.raises(SystemExit) as result:
        renderer.main(arguments(f"--override={override}"))
    assert result.value.code == 2


def test_duplicate_seeds_fail_before_render():
    args = arguments()
    args[args.index("102")] = "101"
    with pytest.raises(SystemExit) as result:
        renderer.main(args)
    assert result.value.code == 2
