"""Render the fourteen missing paired-seed fits as six shared-GPU Jobs."""

import argparse
import contextlib
import io
import json
import math
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.ml import estimate_training_batch_memory as memory
from scripts.ml.run_training_batch import validate_spec


ROOT = Path(__file__).resolve().parents[2]
MASKS = {"cm5": "cancer_mask_5", "ean5": "epithel_and_cancer_5"}
RECIPES = {
    "mean_regression": ("multiscale-mean-linear", [1, 2]),
    "projected_mean_regression": ("multiscale-projected-mean", [0, 1, 2]),
    "slide_attention_regression": ("multiscale-slide-attention", [1, 2]),
}
# Aggregate artifact sizes measured on 2026-11-23, train + val + test.
DOWNLOAD_GIB = {"cm5": 207.98, "ean5": 176.42}


def plan(args: argparse.Namespace) -> list[dict]:
    jobs = []
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        for mask, family in MASKS.items():
            for recipe, (architecture, seeds) in RECIPES.items():
                experiment = f"ml/multiscale/screen/{mask}_{recipe}"
                config = compose(
                    config_name="ml",
                    overrides=[
                        f"+experiment={experiment}",
                        "test_after_fit=true",
                        "trainer.accelerator=gpu",
                        "trainer.devices=1",
                    ],
                )
                resolved = OmegaConf.to_container(
                    config, resolve=True, throw_on_missing=True
                )
                estimate = memory.estimate(
                    argparse.Namespace(
                        architecture=architecture,
                        level=4,
                        mask=family,
                        train_batch_size=1,
                        eval_batch_size=1,
                        train_tile_cap=0,
                        include_test=True,
                        device_gib=args.device_gib,
                    )
                )
                parallel = min(args.max_parallel, len(seeds))
                if parallel > estimate["vram_fit_count_ceiling"]:
                    raise ValueError(
                        f"{experiment} exceeds the 80% VRAM planning budget"
                    )
                # Per fit: downloaded artifacts plus 10% working space. Add one
                # shared allowance for the environment, logs and checkout.
                scratch = math.ceil(DOWNLOAD_GIB[mask] * parallel * 1.1 + 12)
                short_recipe = recipe.removesuffix("_regression").replace("_", "-")
                name = f"ms-{mask}-{short_recipe}-{args.batch_id}"
                jobs.append(
                    {
                        "job_name": name,
                        "mask": mask,
                        "recipe": recipe,
                        "experiment": experiment,
                        "seeds": seeds,
                        "max_parallel": parallel,
                        "gpu": getattr(args, "gpu", "A40"),
                        "cpu": 4 * parallel,
                        "memory": f"{16 * parallel}Gi",
                        "scratch_size": f"{scratch}Gi",
                        "ephemeral_storage": f"{scratch + 8}Gi",
                        "estimate": estimate,
                        "configuration": {
                            key: resolved[key]
                            for key in (
                                "ml",
                                "datamodule",
                                "pyramid",
                                "trainer",
                                "label_mode",
                            )
                        },
                    }
                )
    return jobs


def group_by_mask(
    manifests: list[dict], jobs: list[dict], args: argparse.Namespace
) -> list[dict]:
    grouped = []
    for mask in MASKS:
        members = [
            (manifest, job)
            for manifest, job in zip(manifests, jobs, strict=True)
            if job["mask"] == mask
        ]
        manifest = members[0][0]
        largest_fit = max(job["estimate"]["planning_gib_per_fit"] for _, job in members)
        if (
            largest_fit * args.max_parallel
            > args.device_gib * memory.DEVICE_BUDGET_FRACTION
        ):
            raise ValueError("Grouped fits exceed the 80% VRAM planning budget")
        name = f"ms-{mask}-{args.batch_id}"
        runs = []
        for original, job in members:
            spec = json.loads(
                original["spec"]["template"]["spec"]["containers"][0]["args"][
                    3
                ].replace("$$", "$")
            )
            for run in spec["runs"]:
                runs.append(
                    {
                        "id": f"{job['recipe'].removesuffix('_regression').replace('_', '-')}-{run['id']}",
                        "argv": run["argv"],
                    }
                )
        spec = validate_spec(
            {
                "version": 1,
                "batch_id": name,
                "max_parallel": args.max_parallel,
                "runs": runs,
            }
        )
        manifest["metadata"]["name"] = name
        manifest["metadata"]["annotations"]["mammaprint.rationai.org/batch-id"] = name
        template = manifest["spec"]["template"]
        template["metadata"]["labels"]["app"] = name
        container = template["spec"]["containers"][0]
        scratch = math.ceil(DOWNLOAD_GIB[mask] * args.max_parallel * 1.1 + 12)
        for resources in container["resources"].values():
            resources.update(
                cpu=str(4 * args.max_parallel), memory=f"{16 * args.max_parallel}Gi"
            )
            resources["ephemeral-storage"] = f"{scratch + 8}Gi"
        for volume in template["spec"]["volumes"]:
            if volume["name"] == "scratch":
                volume["emptyDir"]["sizeLimit"] = f"{scratch}Gi"
        container["args"][3] = json.dumps(spec, separators=(",", ":")).replace(
            "$", "$$"
        )
        container["args"][6] = name
        grouped.append(manifest)
    return grouped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", choices=("A40", "mig-2g.24gb"), default="A40")
    parser.add_argument("--device-gib", type=float)
    parser.add_argument("--max-parallel", type=int, choices=range(1, 8), default=2)
    parser.add_argument(
        "--group-by-mask",
        action="store_true",
        help="Queue all seven fits for each mask on one GPU",
    )
    parser.add_argument("--cpu", type=memory.positive_int)
    parser.add_argument("--memory-gib", type=memory.positive_int)
    parser.add_argument("--threads-per-fit", type=memory.positive_int, default=4)
    parser.add_argument(
        "--unlimited-storage",
        action="store_true",
        help="Omit ephemeral-storage resources and emptyDir size limits",
    )
    parser.add_argument(
        "--output-root", default="/mnt/projects/mammaprint/training-batches"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create the reviewed Jobs after explicit approval",
    )
    args = parser.parse_args(argv)
    if args.device_gib is None:
        args.device_gib = 23.0 if args.gpu == "mig-2g.24gb" else 45.0
    if not math.isfinite(args.device_gib) or args.device_gib <= 0:
        parser.error("--device-gib must be finite and positive")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != args.source_commit
    ):
        parser.error("--source-commit must match the checked-out HEAD")
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "ml",
            "configs",
            "scripts/ml",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        text=True,
    )
    if dirty.strip():
        parser.error("Commit the worker, configurations and launcher before rendering")
    # Reuse the established renderer's override validation, isolation, GPU
    # selector, credentials and resource telemetry for every seed batch.
    from scripts.ml import render_training_batch_job as batch

    jobs = plan(args)
    manifests = []
    for job in jobs:
        flags = [
            "--context",
            args.context,
            "--namespace",
            args.namespace,
            "--job-name",
            job["job_name"],
            "--batch-id",
            job["job_name"],
            "--source-commit",
            args.source_commit,
            "--image",
            "cerit.io/rationai/base:2.0.6",
            "--gpu",
            job["gpu"],
            "--cpu",
            str(job["cpu"]),
            "--memory",
            job["memory"],
            "--shm",
            "1Gi",
            "--scratch-size",
            job["scratch_size"],
            "--ephemeral-storage",
            job["ephemeral_storage"],
            "--threads-per-fit",
            str(args.threads_per_fit),
            "--output-root",
            args.output_root,
            "--experiment",
            job["experiment"],
            "--seeds",
            *map(str, job["seeds"]),
            "--max-parallel",
            str(job["max_parallel"]),
            "--override",
            "test_after_fit=true",
            "--override",
            f'+logger.tags.comparison_id="{args.batch_id}"',
            "--override",
            f"+logger.tags.multiscale_screen={job['recipe']}",
            "--override",
            f"+logger.tags.data_variant={job['mask']}",
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            batch.main(flags)
        manifests.append(json.loads(output.getvalue()))
    if args.group_by_mask:
        manifests = group_by_mask(manifests, jobs, args)
    for manifest in manifests:
        pod = manifest["spec"]["template"]["spec"]
        container = pod["containers"][0]
        for resources in container["resources"].values():
            if args.cpu:
                resources["cpu"] = str(args.cpu)
            if args.memory_gib:
                resources["memory"] = f"{args.memory_gib}Gi"
            if args.unlimited_storage:
                resources.pop("ephemeral-storage", None)
        if args.unlimited_storage:
            for volume in pod["volumes"]:
                if "emptyDir" in volume:
                    volume["emptyDir"].pop("sizeLimit", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for manifest in manifests:
        path = args.output_dir / (manifest["metadata"]["name"] + ".json")
        content = json.dumps(manifest, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Different manifest already exists at {path}")
        path.write_text(content)
        paths.append(path)
    (args.output_dir / "comparison-plan.json").write_text(
        json.dumps(
            {
                "configurations": jobs,
                "submission": {
                    "group_by_mask": args.group_by_mask,
                    "cpu_override": args.cpu,
                    "memory_gib_override": args.memory_gib,
                    "threads_per_fit": args.threads_per_fit,
                    "max_parallel": args.max_parallel,
                    "unlimited_storage": args.unlimited_storage,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "fits": sum(len(j["seeds"]) for j in jobs),
                "jobs": len(manifests),
                "manifests": list(map(str, paths)),
            },
            indent=2,
        )
    )
    if args.submit:
        command = [
            "kubectl",
            "--context",
            args.context,
            "--namespace",
            args.namespace,
            "create",
        ]
        for path in paths:
            subprocess.run([*command, "-f", str(path)], check=True)


if __name__ == "__main__":
    main()
