"""Build and attach the RationAI pathologist report for a prediction run."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from mlflow import MlflowClient


DEFAULT_MLFLOW_UI_URI = "https://mlflow.rationai.cloud.trusted.e-infra.cz/"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "report"


def prediction_run_url(ui_uri: str, experiment_id: str, run_id: str) -> str:
    """Return the browser-visible MLflow page for one prediction run."""
    return (
        f"{ui_uri.rstrip('/')}/#/experiments/{quote(experiment_id, safe='')}"
        f"/runs/{quote(run_id, safe='')}"
    )


def build_report_command(
    *,
    config_dir: Path,
    prediction_run_id: str,
    tracking_uri: str,
    user: str,
) -> list[str]:
    """Construct the external report package's Hydra command."""
    return [
        sys.executable,
        "-m",
        "report",
        "--config-dir",
        str(config_dir.resolve()),
        "+reporter=mammaprint",
        f"user={json.dumps(user)}",
        f"mlflow.tracking_uri={json.dumps(tracking_uri)}",
        "metadata.experiment_name=MammaPrint",
        "metadata.run_name=MammaPrint pathologist report",
        "metadata.description=Pathologist review of local tile predictions",
        f"++reporter.evaluation_runs={{{prediction_run_id}:"
        "{artifact_file:predictions/predictions.json,"
        "slide_item_key:report_item_id}}",
    ]


def build_pathologist_report(
    *,
    client: MlflowClient,
    prediction_run_id: str,
    tracking_uri: str,
    user: str,
    title: str | None = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    mlflow_ui_uri: str = DEFAULT_MLFLOW_UI_URI,
    dry_run: bool = False,
) -> int:
    """Run the shared RationAI reporter and attach ``report/report.html``."""
    run = client.get_run(prediction_run_id)
    run_name = run.data.tags.get("mlflow.runName", prediction_run_id)
    environment = os.environ.copy()
    environment.update(
        {
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MAMMAPRINT_PREDICTION_RUN_ID": prediction_run_id,
            "MAMMAPRINT_PREDICTION_ARTIFACT_URI": run.info.artifact_uri,
            "MAMMAPRINT_REPORT_TITLE": title
            or f"MammaPrint tile predictions — {run_name}",
        }
    )
    command = build_report_command(
        config_dir=config_dir,
        prediction_run_id=prediction_run_id,
        tracking_uri=tracking_uri,
        user=user,
    )
    print("Report command:")
    print("  " + " ".join(command))
    if dry_run:
        return 0

    result = subprocess.run(command, check=False, env=environment)
    if result.returncode == 0:
        url = prediction_run_url(
            mlflow_ui_uri, run.info.experiment_id, prediction_run_id
        )
        print(f"PATHOLOGIST_REPORT_URL={url}")
        print("PATHOLOGIST_REPORT_ARTIFACT=report/report.html")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-run-id", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        required=os.getenv("MLFLOW_TRACKING_URI") is None,
    )
    parser.add_argument(
        "--mlflow-ui-uri",
        default=os.getenv("MLFLOW_UI_URI", DEFAULT_MLFLOW_UI_URI),
    )
    parser.add_argument("--user", default=os.getenv("USER", "unknown"))
    parser.add_argument("--title")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    return build_pathologist_report(
        client=client,
        prediction_run_id=args.prediction_run_id,
        tracking_uri=args.tracking_uri,
        user=args.user,
        title=args.title,
        config_dir=args.config_dir,
        mlflow_ui_uri=args.mlflow_ui_uri,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
