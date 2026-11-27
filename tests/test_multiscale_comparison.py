import argparse
import json
import subprocess

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.ml import run_multiscale_comparison as comparison


def arguments(**overrides):
    return argparse.Namespace(
        device_gib=45.0, max_parallel=2, batch_id="compare-v1", **overrides
    )


def test_missing_seed_matrix_and_shared_gpu_resources():
    jobs = comparison.plan(arguments())
    assert len(jobs) == 6
    assert sum(len(job["seeds"]) for job in jobs) == 14
    for job in jobs:
        projected = "projected_mean" in job["experiment"]
        assert job["seeds"] == ([0, 1, 2] if projected else [1, 2])
        assert job["max_parallel"] == 2
        assert job["cpu"] == 8 and job["memory"] == "32Gi"
        assert job["estimate"]["planning_gib_per_fit"] * 2 <= 45 * 0.8
        scratch = int(job["scratch_size"].removesuffix("Gi"))
        assert scratch > 2 * comparison.DOWNLOAD_GIB[job["job_name"].split("-")[1]]
        assert int(job["ephemeral_storage"].removesuffix("Gi")) > scratch
        cfg = job["configuration"]
        assert cfg["label_mode"] == "index"
        assert cfg["pyramid"]["level_order"] == [4, 3, 2]
        assert (
            cfg["trainer"]["callbacks"]["model_checkpoint"]["monitor"] == "valid/loss"
        )


def test_device_too_small_rejects_instead_of_overcommitting():
    args = arguments()
    args.device_gib = 2
    with pytest.raises(ValueError, match="VRAM"):
        comparison.plan(args)


@pytest.mark.parametrize(
    "recipe,parameters,width",
    [
        ("mean_regression", 7681, 7680),
        ("projected_mean_regression", 671745, 768),
        ("slide_attention_regression", 1067266, 768),
    ],
)
def test_composed_models_match_historical_checkpoint_shapes(recipe, parameters, width):
    with initialize_config_dir(
        version_base=None, config_dir=str(comparison.ROOT / "configs")
    ):
        config = compose(
            config_name="ml",
            overrides=[f"+experiment=ml/multiscale/screen/cm5_{recipe}"],
        )
        module = instantiate(config.ml)
        assert (
            sum(p.numel() for p in module.parameters() if p.requires_grad) == parameters
        )
        assert config.ml.head.in_dim == width
        assert set(
            map(
                int,
                OmegaConf.to_container(config.datamodule.train.levels, resolve=True),
            )
        ) == {2, 3, 4}


def test_single_fit_mode_scales_host_and_disk_requests():
    args = arguments()
    args.max_parallel = 1
    jobs = comparison.plan(args)
    assert all(j["cpu"] == 4 and j["memory"] == "16Gi" for j in jobs)
    assert sum(len(j["seeds"]) for j in jobs) == 14


@pytest.mark.parametrize(
    "grouped,parallel,unlimited",
    [(False, 2, False), (True, 2, True), (True, 7, True), (True, 7, False)],
)
def test_full_renderer_is_offline_and_gives_distinct_fit_names(
    tmp_path, monkeypatch, grouped, parallel, unlimited
):
    original = subprocess.check_output

    def git_read(command, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", git_read)
    revision = original(["git", "rev-parse", "HEAD"], text=True).strip()
    flags = [
        "--context",
        "kubas-cluster",
        "--namespace",
        "rationai-jobs-ns",
        "--batch-id",
        "test-v1",
        "--source-commit",
        revision,
        "--output-dir",
        str(tmp_path),
    ]
    if grouped:
        flags += [
            "--group-by-mask",
            "--cpu",
            "16",
            "--memory-gib",
            "64",
            "--threads-per-fit",
            "2" if parallel == 7 else "8",
            "--max-parallel",
            str(parallel),
        ]
    if unlimited:
        flags.append("--unlimited-storage")
    comparison.main(flags)
    names = set()
    fits = 0
    for path in tmp_path.glob("ms-*.json"):
        manifest = json.loads(path.read_text())
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        spec = json.loads(container["args"][3].replace("$$", "$"))
        assert spec["max_parallel"] == parallel
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
        if grouped:
            assert len(spec["runs"]) == 7
            assert (
                container["args"][6] == manifest["metadata"]["name"] == spec["batch_id"]
            )
            for resources in container["resources"].values():
                expected = {"cpu": "16", "memory": "64Gi", "nvidia.com/gpu": 1}
                if not unlimited:
                    expected["ephemeral-storage"] = resources["ephemeral-storage"]
                    assert (
                        int(resources["ephemeral-storage"].removesuffix("Gi")) > 7 * 176
                    )
                assert resources == expected
            if unlimited:
                assert all(
                    "sizeLimit" not in v.get("emptyDir", {})
                    for v in manifest["spec"]["template"]["spec"]["volumes"]
                )
        for run in spec["runs"]:
            assert "test_after_fit=true" in run["argv"]
            assert '+logger.tags.comparison_id="test-v1"' in run["argv"]
            names.add((spec["batch_id"], run["id"]))
            fits += 1
    assert fits == len(names) == 14
    assert len(list(tmp_path.glob("ms-*.json"))) == (2 if grouped else 6)
