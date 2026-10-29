"""Publish the standard RationAI report for an existing explanation run (no inference)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow

from ml.explainability.reporting import (
    DEFAULT_REPORT_METHODS,
    REPORT_METHODS,
    write_rationai_report,
)
from ml.explainability.viewer import DEFAULT_XOPAT_UI_URI, build_xopat_report_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tracking-uri", default="https://mlflow-s3.rationai.cloud.trusted.e-infra.cz"
    )
    parser.add_argument("--mlflow-ui-uri", help="Defaults to the tracking URI.")
    parser.add_argument("--xopat-ui-uri", default=DEFAULT_XOPAT_UI_URI)
    parser.add_argument("--artifact-mount", default="secure_mlflow")
    parser.add_argument("--artifact-path", default="explanations")
    parser.add_argument("--report-artifact-path", default="report")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=REPORT_METHODS,
        default=list(DEFAULT_REPORT_METHODS),
        help="Methods to compare (default: tile removal, single-tile output, attention).",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Render locally for inspection."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    root = args.artifact_path.strip("/")
    output = (
        args.output_dir or Path("outputs/explanation-reports") / args.run_id
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Download only two small metadata files, not embeddings or mask images.
    manifest_path = Path(
        client.download_artifacts(args.run_id, f"{root}/manifest.json", str(output))
    )
    slides_path = Path(
        client.download_artifacts(args.run_id, f"{root}/slides.parquet", str(output))
    )
    manifest = json.loads(manifest_path.read_text())
    masks = {
        file.path
        for target in manifest["targets"]
        for method in args.methods
        if method in manifest["methods"]
        for file in client.list_artifacts(
            args.run_id, f"{root}/masks/{method}/{target}"
        )
        if not file.is_dir
    }
    report = write_rationai_report(
        output / "report.html",
        slides_path=slides_path,
        manifest_path=manifest_path,
        run_id=args.run_id,
        experiment_id=run.info.experiment_id,
        artifact_root=root,
        slide_key_artifact=f"{root}/slide_keys.json",
        xopat_ui_uri=args.xopat_ui_uri,
        mlflow_ui_uri=args.mlflow_ui_uri or args.tracking_uri,
        artifact_mount=args.artifact_mount,
        mask_artifact_paths=masks,
        included_methods=args.methods,
    )
    print(f"REPORT_FILE={report}")
    if args.no_upload:
        return 0
    client.log_artifact(
        args.run_id, str(report), artifact_path=args.report_artifact_path
    )
    url = build_xopat_report_url(
        xopat_ui_uri=args.xopat_ui_uri,
        experiment_id=run.info.experiment_id,
        run_id=args.run_id,
        artifact_path=args.report_artifact_path,
        artifact_mount=args.artifact_mount,
    )
    for key, value in {
        "report.artifact": f"{args.report_artifact_path.strip('/')}/report.html",
        "report.template": "RationAI report_template.html",
        "report.methods": json.dumps(
            [method for method in args.methods if method in manifest["methods"]]
        ),
        "report.status": "ready",
        "pathologist_review_url": url,
    }.items():
        client.set_tag(args.run_id, key, value)
    print(f"PATHOLOGIST_REVIEW_URL={url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
