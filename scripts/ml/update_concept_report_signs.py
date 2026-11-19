"""Prepare or apply a presentation-only repair to a finished MLflow report."""

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.concept_explainability.presentation import (
    EFFECT_DISPLAY_VERSION,
    upgrade_effect_html,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_report(client, run_id, experiment_id, work_dir, *, apply=False):
    """Download, validate, back up, replace and read back one HTML artifact."""
    import pandas as pd

    uri = urlsplit(client.tracking_uri)
    if (
        uri.scheme not in {"http", "https"}
        or not uri.hostname
        or any((uri.username, uri.password, uri.query, uri.fragment))
    ):
        raise ValueError("Use an explicit credentials-free HTTP(S) tracking URI.")
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Expected an exact MLflow run ID.")
    run = client.get_run(run_id)
    if (
        str(run.info.experiment_id) != str(experiment_id)
        or run.info.status != "FINISHED"
        or run.info.lifecycle_stage != "active"
    ):
        raise ValueError("Expected an active FINISHED run in the explicit experiment.")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=False)

    def download(artifact, folder):
        dest = root / folder
        dest.mkdir(parents=True, exist_ok=True)
        return Path(client.download_artifacts(run_id, artifact, str(dest)))

    manifest_path = download("val/report_manifest.json", "original")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_run_id"] == run_id:
        raise ValueError("Refusing to modify a source model run.")
    k = manifest["n_components"]
    if type(k) is not int or k < 1 or manifest["split"] != "val":
        raise ValueError("Expected a validation concept report manifest.")
    artifact = f"val/naming_k{k}.html"
    original = download(artifact, "original")
    raw = download("val/concept_effects.parquet", "original")
    effects = pd.read_parquet(raw)
    if set(effects.concept) != set(range(k)):
        raise ValueError("Effect concepts do not match the report manifest.")
    before = original.read_text()
    report_id_match = re.search(r"const REPORT_ID=(\{[^;]+\});", before)
    if report_id_match is None:
        raise ValueError("Report identity is missing.")
    report_id = json.loads(report_id_match[1])
    for key in ("source_run_id", "dictionary_id", "split"):
        if report_id[key] != manifest[key]:
            raise ValueError(f"Report and manifest disagree on {key}.")
    after = upgrade_effect_html(before, effects)
    updated = root / "updated" / original.name
    updated.parent.mkdir()
    updated.write_text(after)
    receipt = {
        "run_id": run_id,
        "experiment_id": str(experiment_id),
        "tracking_uri": client.tracking_uri,
        "artifact": artifact,
        "display_version": EFFECT_DISPLAY_VERSION,
        "original_sha256": digest(original),
        "updated_sha256": digest(updated),
        "raw_effects_sha256": digest(raw),
        "original_manifest_sha256": digest(manifest_path),
        "presentation_code_sha256": digest(
            Path(__file__).resolve().parents[2]
            / "ml/concept_explainability/presentation.py"
        ),
        "prepared_at": datetime.now(UTC).isoformat(),
        "status": "already_updated" if before == after else "prepared",
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if not apply:
        return receipt

    archive = f"report_updates/{EFFECT_DISPLAY_VERSION}/val"
    backup_dir = f"{archive}/original"
    backup_artifact = f"{backup_dir}/{original.name}"
    if before == after:
        # Complete the audit if a previous attempt replaced the HTML but failed
        # while publishing its final verified receipt.
        previous = json.loads(
            download(f"{archive}/receipt.json", "previous-receipt").read_text()
        )
        if (
            any(
                previous.get(key) != receipt[key]
                for key in (
                    "run_id",
                    "experiment_id",
                    "artifact",
                    "display_version",
                    "updated_sha256",
                    "raw_effects_sha256",
                    "original_manifest_sha256",
                )
            )
            or previous.get("backup_artifact") != backup_artifact
        ):
            raise ValueError(
                "Updated report does not match its previous repair receipt."
            )
        backup = download(backup_artifact, "backup-check")
        if digest(backup) != previous["original_sha256"]:
            raise ValueError("Original backup does not match the repair receipt.")
        if previous["status"] == "verified":
            return {**previous, "status": "already_updated"}
        receipt = previous
        receipt["status"] = "verified"
    else:
        _replace_report(
            client,
            run_id,
            download,
            original,
            updated,
            raw,
            manifest_path,
            artifact,
            backup_dir,
            backup_artifact,
            archive,
            receipt,
            receipt_path,
        )
        receipt.update(status="verified", backup_artifact=backup_artifact)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    client.log_artifact(run_id, str(receipt_path), archive)
    if digest(download(f"{archive}/receipt.json", "receipt-check")) != digest(
        receipt_path
    ):
        raise ValueError("HTML updated but receipt read-back failed.")
    return receipt


def _replace_report(
    client,
    run_id,
    download,
    original,
    updated,
    raw,
    manifest_path,
    artifact,
    backup_dir,
    backup_artifact,
    archive,
    receipt,
    receipt_path,
):
    existing = client.list_artifacts(run_id, backup_dir)
    if any(item.path == backup_artifact for item in existing):
        backup = download(backup_artifact, "backup-check")
        if digest(backup) != digest(original):
            raise ValueError(
                "A different original backup already exists; refusing overwrite."
            )
    else:
        client.log_artifact(run_id, str(original), backup_dir)
        backup = download(backup_artifact, "backup-check")
        if digest(backup) != digest(original):
            raise ValueError(
                "Backup verification failed; original HTML was not replaced."
            )
    # Detect intervening report/effect changes before overwriting anything in val.
    if (
        digest(download(artifact, "current")) != digest(original)
        or digest(download("val/concept_effects.parquet", "current")) != digest(raw)
        or digest(download("val/report_manifest.json", "current"))
        != digest(manifest_path)
    ):
        raise ValueError(
            "Source artifacts changed during preparation; refusing replacement."
        )
    receipt["backup_artifact"] = backup_artifact
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    client.log_artifact(run_id, str(receipt_path), archive)
    if digest(download(f"{archive}/receipt.json", "prepared-receipt-check")) != digest(
        receipt_path
    ):
        raise ValueError(
            "Prepared receipt verification failed; original HTML was not replaced."
        )
    client.log_artifact(run_id, str(updated), "val")
    if digest(download(artifact, "verified")) != digest(updated):
        raise ValueError(
            f"Replacement read-back failed; original is backed up at {backup_artifact}."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--work-dir", required=True, type=Path, help="Fresh local directory"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and replace the HTML in the same run",
    )
    args = parser.parse_args(argv)
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    print(
        json.dumps(
            update_report(
                client, args.run_id, args.experiment_id, args.work_dir, apply=args.apply
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
