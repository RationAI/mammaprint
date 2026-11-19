"""Preview or submit one of the two fixed L3 mean-model explanation jobs.

The default only prints the job specification. Use --submit to create a job.
Pass --slide-id for a one-slide smoke run before the full test split.
"""

from __future__ import annotations

import argparse
import re

from scripts.ml.submit_explain import build_job_spec, build_parser, submit


MODELS = {
    "regression": {
        "run": "656ef0c37f264d6eab1d15ed20867335",
        "checkpoint": "aa8f3c75a76bcb48234df759b9c8ae20f36ff5028b354231539b7a23056a4883",
        "config": "da8a40dd99feebfe70573aeff31e64f0a49e0cdf19f3c3498432ed12b577419d",
    },
    "bce": {
        "run": "82d77ce5ee274ee9b8853a4b2abcc33e",
        "checkpoint": "74a50ce0c5565a671d00f189dc5bb25a8e7f45f62101ad72e0a6446536e74fc1",
        "config": "3a302e6fc07c0e1ceacbf0caa28d87c2523cfaf1d1d3271e79fc30baf3e4f95e",
    },
}


def job_spec(model: str, revision: str, slide_id: str | None = None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be the full pushed commit SHA.")
    source = MODELS[model]
    arguments = [
        "--checkpoint-uri",
        f"mlflow-artifacts:/3/{source['run']}/artifacts/checkpoints/best",
        "--branch",
        revision,
        "--gpu",
        "none",
        "--cpu",
        "4",
        "--memory",
        "8Gi",
        "--device",
        "cpu",
        "--resolved-source-config",
        "--expected-checkpoint-sha256",
        source["checkpoint"],
        "--expected-source-config-sha256",
        source["config"],
        "--scratch-root",
        "/mnt/projects/mammaprint/exact-mean-explanations",
        "--run-name",
        f"exact-mean-{model}-{'smoke' if slide_id else 'test'}-{revision[:8]}",
    ]
    for override in (
        "explain.exact_mean.enabled=true",
        "explain.ig.enabled=false",
        "explain.attention.enabled=false",
        "explain.single.enabled=true",
        "explain.leave_one_out.enabled=true",
        "explain.patch_flipping.enabled=false",
    ):
        arguments.extend(("--override", override))
    if slide_id:
        arguments.extend(("--slide-id", slide_id))
    return build_job_spec(build_parser().parse_args(arguments))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--slide-id", help="Restrict to one test slide for a smoke run."
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = job_spec(args.model, args.revision, args.slide_id)
    print(spec.printable())
    if args.submit:
        submit(spec)


if __name__ == "__main__":
    main()
