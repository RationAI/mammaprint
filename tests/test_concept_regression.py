"""Scalar-index adapter, additive accounting and bounded native workflow checks."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from test_concept_evaluation import fixture_source, local_download
from test_concept_metric import metric_source

from ml.concept_explainability import metric
from ml.concept_explainability.evaluation import (
    evaluate_bag,
    evaluate_concepts,
    fit_dictionary,
    linear_index_decomposition,
    load_dictionary,
)
from ml.concept_explainability.model import model_outputs
from ml.concept_explainability.report import build_report, render_naming_html
from ml.concept_explainability.upstream import ConceptAutoencoder
from ml.models.aggregators import MeanPool
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule


def test_scalar_metric_matches_upstream_output_algebra_without_centering():
    z = np.random.default_rng(42).normal(size=(30, 4)).astype(np.float32)
    w = np.array([[0.2, -0.7, 0.1, 1.1]], dtype=np.float32)
    b = np.array([0.3], dtype=np.float32)
    output = z.astype(np.float64) @ w.T + b
    expected = np.outer(w[0], w[0]) / output.var()
    expected *= 4 / np.trace(expected)
    expected += 0.3 * np.eye(4)
    transform = metric.regression_output_metric(z, w, b)
    np.testing.assert_allclose(transform.T @ transform, expected, atol=5e-7)
    classifier = metric.classifier_output_metric(z, w, b)
    np.testing.assert_allclose(
        transform.T @ transform, classifier.T @ classifier, atol=5e-7
    )
    for bad in [np.ones_like(z), np.full_like(z, np.nan), z[:1]]:
        with pytest.raises(ValueError):
            metric.regression_output_metric(bad, w, b)
    with pytest.raises(ValueError, match="variance"):
        metric.regression_output_metric(z, np.zeros_like(w), b)


def test_index_preflight_is_frozen_and_rejects_other_graphs_before_data(tmp_path):
    source, original = metric_source(tmp_path, label_mode="index")
    module = metric.preflight_metric_source(source)
    assert not module.training and all(not p.requires_grad for p in module.parameters())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, module.state_dict()[key])
    mutations = [
        ("label_mode", "both"),
        ("ml.loss._target_", "torch.nn.BCEWithLogitsLoss"),
        ("ml.aggregator._target_", "ml.models.aggregators.AttentionMIL"),
        ("ml.head._target_", "ml.models.heads.MLPHead"),
        ("ml.output_activation", {"_target_": "torch.nn.Tanh"}),
        ("datamodule.val.label_mode", "type"),
    ]
    for key, value in mutations:
        bad = copy.copy(source)
        bad.config = OmegaConf.create(OmegaConf.to_container(source.config))
        OmegaConf.update(bad.config, key, value)
        with (
            patch(
                "ml.concept_explainability.evaluation.source_dataset",
                side_effect=AssertionError("downloaded"),
            ),
            pytest.raises(ValueError, match="metric"),
        ):
            fit_dictionary(
                bad,
                OmegaConf.create({"n_components": 2, "metric": {"enabled": True}}),
                tmp_path / "bad",
                use_ray=False,
            )


def test_signed_contributions_close_unbounded_native_prediction_with_residual():
    module = MammaprintModule(
        IdentityEncoder(3), MeanPool(3), LinearHead(3), loss=torch.nn.MSELoss()
    ).eval()
    with torch.no_grad():
        module.head.linear.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]]))
        module.head.linear.bias.fill_(0.3)
    dictionary = ConceptAutoencoder(
        np.array([[1, 0, 0], [0, 1, 0]], np.float32),
        np.array([0.2, 0.4, -0.1], np.float32),
        False,
        10,
    )
    codes = np.array([[1, 3], [2, 1]], np.float32)
    reconstructed = dictionary.decode(codes)
    features = reconstructed + np.array([0, 0, 0.4], np.float32)
    original = float(model_outputs(module, torch.from_numpy(features), "index")[0])
    rebuilt = float(model_outputs(module, torch.from_numpy(reconstructed), "index")[0])
    contributions, summary = linear_index_decomposition(
        module, dictionary, codes, original, rebuilt
    )
    np.testing.assert_allclose(contributions, [3, -2], atol=1e-7)
    assert summary["baseline"] == pytest.approx(0.25)
    assert summary["residual"] == pytest.approx(0.2, abs=1e-6)
    assert summary["baseline"] + contributions.sum() + summary[
        "residual"
    ] == pytest.approx(original)
    assert original > 1  # Neither sigmoid nor clipping is part of this regressor.
    repeated, _ = linear_index_decomposition(
        module, dictionary, np.tile(codes, (3, 1)), original, rebuilt
    )
    np.testing.assert_array_equal(repeated, contributions)
    slide = type("Slide", (), {"embeddings": features})()
    effects = evaluate_bag(
        module, slide, dictionary, np.array([0.2, 0.2]), "index", "cpu", codes=codes
    )
    assert effects["add"][0, 0] > 0 and effects["add"][1, 0] < 0
    assert effects["subtract"][1, 0] < 0
    module.train()
    with pytest.raises(ValueError, match="frozen"):
        linear_index_decomposition(module, dictionary, codes, original, rebuilt)


@patch(
    "ml.data.datasets.single_scale.download_level_sources", side_effect=local_download
)
def test_native_regression_fit_evaluate_and_report_are_bound_and_blinded(
    _download, tmp_path
):
    source = fixture_source(tmp_path, label_mode="index")
    # Native generic TIFF fixtures have scanner level zero, not the real cohort's level 3.
    card = source.config.dataset.levels.pop("3")
    source.config.dataset.levels["0"] = card
    for split in ("train", "val", "test"):
        for i in range(2):
            Image.new("RGB", (512, 512), (100, 50, 150)).save(
                tmp_path / f"{split}{i}.tiff"
            )
    native, _ = metric_source(tmp_path, label_mode="index")
    source.config.ml = native.config.ml
    source.checkpoint_path, source.checkpoint_sha256 = (
        native.checkpoint_path,
        native.checkpoint_sha256,
    )
    original_bytes = source.checkpoint_path.read_bytes()
    settings = OmegaConf.create(
        {
            "n_components": 2,
            "metric": {
                "enabled": True,
                "n_tiles": 20000,
                "ridge": 0.3,
                "attention_weight": 1,
            },
            "nmf": {
                "batch_size": 32,
                "shuffle_buffer_size": 32,
                "random_state": 42,
                "epochs": 1,
                "nnls_iter": 10,
                "gauge_fix": True,
            },
            "semi_nmf": {"forget": 0.95, "ridge": 1e-6},
        }
    )
    with patch(
        "ml.concept_explainability.evaluation.classifier_output_metric",
        side_effect=AssertionError("classifier metric used"),
    ):
        fit_dictionary(source, settings, tmp_path / "dictionary", use_ray=False)
    assert set(source.datasets) == {"train"}
    dictionary = load_dictionary(tmp_path / "dictionary", source=source)
    train = np.concatenate(
        [
            pd.read_parquet(p).embedding.to_list()
            for p in sorted((tmp_path / "train").glob("*.parquet"))
        ]
    )
    np.testing.assert_allclose(
        dictionary.mu, train.astype(np.float64).mean(0).astype(np.float32)
    )
    fitted = json.loads((tmp_path / "dictionary/manifest.json").read_text())
    assert fitted["metric"]["adapter"] == metric.REGRESSION_METRIC_ADAPTER
    assert (
        fitted["metric"]["output_terms"] == 1
        and "binary_logits" not in fitted["metric"]
    )
    assert fitted["metric"]["sample"]["split"] == "train"
    wrong = copy.copy(source)
    wrong.config = OmegaConf.create(OmegaConf.to_container(source.config))
    wrong.config.label_mode = "type"
    with pytest.raises(ValueError, match="adapter"):
        load_dictionary(tmp_path / "dictionary", source=wrong)
    evaluate_concepts(source, tmp_path / "dictionary", "val", tmp_path / "val-result")
    assert (
        set(source.datasets) == {"train", "val"}
        and source.checkpoint_path.read_bytes() == original_bytes
    )
    result = tmp_path / "val-result"
    manifest = json.loads((result / "manifest.json").read_text())
    effects = pd.read_parquet(result / "concept_effects.parquet")
    assert set(effects.units) == {"index"} and set(effects.group) == {"all"}
    contributions = pd.read_parquet(result / "slide_concept_contributions.parquet")
    slides = pd.read_parquet(result / "slide_index_decomposition.parquet")
    sums = contributions.groupby("slide_id").contribution.sum()
    np.testing.assert_allclose(
        slides.baseline + slides.slide_id.map(sums) + slides.residual,
        slides.prediction,
        atol=1e-9,
    )
    fidelity = pd.read_parquet(result / "fidelity.parquet")
    assert (
        np.isfinite(fidelity.raw_mae).all()
        and np.isfinite(fidelity.residual_abs_p95).all()
    )
    assert fidelity.zero_boundary_agreement.iloc[0] in [0, 0.5, 1]
    page = render_naming_html(
        manifest,
        pd.DataFrame({"concept": [0], "slide_id": ["<unsafe>"], "xopat_url": ["url"]}),
        ["image"],
        effects,
        pd.DataFrame(),
        fidelity,
        contributions=contributions,
        decompositions=slides,
    )
    assert (
        'class="blind"' in page
        and 'class="model-info"><h2>Per-slide index contributions' in page
    )
    assert "Expected index direction" in page and "Expected luminal type" not in page
    assert (
        '"expected_question": "index_direction"' in page
        and "reconstruction error" in page
    )
    assert "<unsafe>" not in page and "&lt;unsafe&gt;" in page
    actual_report = build_report(
        source, tmp_path / "dictionary", result, tmp_path / "report"
    )
    assert (
        actual_report.is_file()
        and "Per-slide index contributions" in actual_report.read_text()
    )
    assert len(list((tmp_path / "report/masks").glob("concept_*"))) == 2
    assert not (tmp_path / "report/.report-codes.npy").exists()
    report_manifest = json.loads((tmp_path / "report/report_manifest.json").read_text())
    assert (
        report_manifest["label_mode"] == "index"
        and "index_decomposition" in report_manifest
    )
    assert set(source.datasets) == {"train", "val"}
    changed = contributions.copy()
    changed.loc[0, "contribution"] += 1
    with pytest.raises(ValueError, match="account"):
        render_naming_html(
            manifest,
            pd.DataFrame(),
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            fidelity,
            contributions=changed,
            decompositions=slides,
        )
