import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.concept_explainability.presentation import (
    EFFECT_NOTE,
    effect_display_frame,
    upgrade_effect_html,
)
from scripts.ml.update_concept_report_signs import update_report


RUN_ID = "d5312ae9136a496fa34bd4324946fe78"


def effects(classifier=False):
    return pd.DataFrame(
        [
            {
                "target": "luminal_a_probability" if classifier else "mammaprint_index",
                "group": group,
                "concept": k,
                "n_slides": 304,
                "add": add,
                "subtract": subtract,
                "effect": (add + subtract) / 2,
                "same_direction": 1.0,
                "disagrees": False,
                "clipped_tiles": 0.171864,
                "units": "percentage_points" if classifier else "index",
                "status": "evaluated",
            }
            for group in (["A", "B"] if classifier else ["all"])
            for k, add, subtract in [(0, -0.041535, -0.039561), (1, 0.02, 0.018)]
        ]
    )


def legacy_page(frame):
    identity = {
        "source_run_id": "source",
        "dictionary_id": "dictionary",
        "split": "val",
        "n_components": 2,
    }
    sections = "".join(
        f'<section><h2>Concept {k}</h2><img src="data:image/jpeg;base64,abc"><a href="https://viewer/#tile">tile</a>'
        f'<div class="model-info">{rows.to_html(index=False, escape=True)}</div></section>'
        for k, rows in frame.groupby("concept")
    )
    return (
        '<html><body class="blind"><button onclick="revealResults()">Show labels and effects</button>'
        "<p>Effects compare two-sided concept changes against reconstructed-model output, not the original embeddings. Legacy text.</p>"
        + sections
        + f"<script>const REPORT_ID={json.dumps(identity)};function revealResults(){{}}</script></body></html>"
    )


@pytest.mark.parametrize("classifier", [False, True])
def test_after_minus_before_and_fixed_probability_target(classifier):
    raw = effects(classifier)
    saved = raw.copy(deep=True)
    display = effect_display_frame(raw)
    factor = np.where(raw.group == "B", -1, 1)
    np.testing.assert_allclose(display["Increase concept: change"], factor * raw["add"])
    np.testing.assert_allclose(
        display["Decrease concept: change"], -factor * raw["subtract"]
    )
    assert (display["Direction consistency (%)"] == 100).all()
    np.testing.assert_allclose(display["Zero-limited tiles (%)"], 17.1864)
    pd.testing.assert_frame_equal(raw, saved)
    old = legacy_page(raw)
    new = upgrade_effect_html(old, raw)
    assert "+0.039561" in new
    assert EFFECT_NOTE in new
    assert "<th>subtract</th>" not in new
    assert old[old.index("<script>") :] == new[new.index("<script>") :]
    assert new.count('src="data:image/jpeg;base64,abc"') == 2
    assert new.count('href="https://viewer/#tile"') == 2
    assert 'class="blind"' in new
    assert upgrade_effect_html(new, raw) == new
    with pytest.raises(ValueError, match="match"):
        upgrade_effect_html(old, raw.assign(add=raw["add"] + 1))
    with pytest.raises(ValueError, match="concepts"):
        upgrade_effect_html(old.replace("Concept 1", "Concept 9"), raw)


class Client:
    tracking_uri = "https://mlflow.example.org"

    def __init__(
        self,
        root,
        *,
        corrupt_backup=False,
        concurrent_change=False,
        fail_final_receipt=False,
    ):
        self.root = root
        self.writes = []
        self.corrupt_backup = corrupt_backup
        self.concurrent_change = concurrent_change
        self.fail_final_receipt = fail_final_receipt

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(
                experiment_id="84", status="FINISHED", lifecycle_stage="active"
            )
        )

    def download_artifacts(self, run_id, artifact, dest):
        path = Path(dest) / Path(artifact).name
        shutil.copyfile(self.root / artifact, path)
        if (self.corrupt_backup and artifact.startswith("report_updates/")) or (
            self.concurrent_change and Path(dest).name == "current"
        ):
            path.write_text("changed")
        return str(path)

    def list_artifacts(self, run_id, path):
        directory = self.root / path
        return [
            SimpleNamespace(path=str(p.relative_to(self.root)))
            for p in directory.glob("*")
        ]

    def log_artifact(self, run_id, path, artifact_path):
        if (
            self.fail_final_receipt
            and Path(path).name == "receipt.json"
            and json.loads(Path(path).read_text())["status"] == "verified"
        ):
            self.fail_final_receipt = False
            raise OSError("Receipt upload interrupted")
        self.writes.append((run_id, artifact_path, Path(path).name))
        dest = self.root / artifact_path / Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / "remote"
    val = root / "val"
    val.mkdir(parents=True)
    frame = effects()
    frame.to_parquet(val / "concept_effects.parquet", index=False)
    (val / "naming_k2.html").write_text(legacy_page(frame))
    (val / "report_manifest.json").write_text(
        json.dumps(
            {
                "n_components": 2,
                "split": "val",
                "source_run_id": "source",
                "dictionary_id": "dictionary",
            }
        )
    )
    return root


def test_prepare_apply_backup_and_idempotent_readback(remote, tmp_path):
    client = Client(remote)
    original = (remote / "val/naming_k2.html").read_bytes()
    raw = (remote / "val/concept_effects.parquet").read_bytes()
    plan = update_report(client, RUN_ID, "84", tmp_path / "plan")
    assert plan["status"] == "prepared"
    assert client.writes == []
    result = update_report(client, RUN_ID, "84", tmp_path / "apply", apply=True)
    assert result["status"] == "verified"
    assert (remote / result["backup_artifact"]).read_bytes() == original
    assert (remote / "val/concept_effects.parquet").read_bytes() == raw
    assert [entry for entry in client.writes if entry[1] == "val"] == [
        (RUN_ID, "val", "naming_k2.html")
    ]
    count = len(client.writes)
    assert (
        update_report(client, RUN_ID, "84", tmp_path / "again", apply=True)["status"]
        == "already_updated"
    )
    assert len(client.writes) == count


@pytest.mark.parametrize("failure", ["corrupt_backup", "concurrent_change"])
def test_refuse_replacement_when_backup_or_current_source_differs(
    remote, tmp_path, failure
):
    client = Client(remote, **{failure: True})
    original = (remote / "val/naming_k2.html").read_bytes()
    with pytest.raises(ValueError):
        update_report(client, RUN_ID, "84", tmp_path / "apply", apply=True)
    assert all(entry[1] != "val" for entry in client.writes)
    assert (remote / "val/naming_k2.html").read_bytes() == original


def test_wrong_experiment_is_rejected_before_download(remote, tmp_path):
    with pytest.raises(ValueError, match="experiment"):
        update_report(Client(remote), RUN_ID, "3", tmp_path / "wrong", apply=True)
    assert not (tmp_path / "wrong").exists()


def test_identical_display_tables_remain_idempotent():
    raw = effects().assign(add=0.0, subtract=0.0, effect=0.0)
    new = upgrade_effect_html(legacy_page(raw), raw)
    assert upgrade_effect_html(new, raw) == new


def test_retry_finishes_receipt_after_html_was_replaced(remote, tmp_path):
    client = Client(remote, fail_final_receipt=True)
    with pytest.raises(OSError, match="interrupted"):
        update_report(client, RUN_ID, "84", tmp_path / "interrupted", apply=True)
    result = update_report(client, RUN_ID, "84", tmp_path / "retry", apply=True)
    assert result["status"] == "verified"
    assert len([entry for entry in client.writes if entry[1] == "val"]) == 1
