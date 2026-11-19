"""Test-only comparison metrics, class orientation, grouping and hook lifetime."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from ml.test_evaluation import (
    ComparisonTestCollector,
    comparison_metrics,
    metric_scalars,
)


def test_weighted_metrics_are_distinct_from_a_positive_binary_f1():
    result = comparison_metrics([2, 1, 3, 4, -1], [1, 1, 0, 0, 0], regression=False)
    m = result["metrics"]
    assert m["f1_a"] == pytest.approx(2 / 3)
    assert m["f1_b"] == pytest.approx(1 / 2)
    assert m["weighted_f1"] == pytest.approx((2 * 2 / 3 + 3 / 2) / 5)
    assert m["weighted_precision"] == pytest.approx(0.8)
    assert m["weighted_recall"] == pytest.approx(0.6)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert "spearman_index_correlation" not in m


def test_ranking_uses_raw_scores_and_reverses_both_labels_and_scores_for_b():
    scores = np.array([0.9, 0.8, -0.1, -0.2])
    targets = np.array([0.5, -0.5, 0.1, -0.1])
    m = comparison_metrics(scores, targets, regression=True)["metrics"]
    assert m["auroc"] == pytest.approx(0.75)
    for name, labels, ranking in (
        ("a", targets >= 0, scores),
        ("b", targets < 0, -scores),
    ):
        p, r, _ = precision_recall_curve(labels, ranking)
        assert m[f"auprc_{name}_trapezoid"] == pytest.approx(auc(r, p))
        assert m[f"average_precision_{name}"] == pytest.approx(
            average_precision_score(labels, ranking)
        )
    assert m["spearman_index_correlation"] == pytest.approx(
        spearmanr(scores, targets).statistic
    )
    assert m["pearson_index_correlation"] == pytest.approx(
        pearsonr(scores, targets).statistic
    )


def test_zero_index_belongs_to_a_and_undefined_metrics_are_not_fabricated():
    report = {
        "slide": comparison_metrics([0, 0], [0, 0], regression=True),
        "patient": None,
    }
    assert report["slide"]["n_a"] == 2
    assert report["slide"]["metrics"]["f1_a"] == 1
    assert report["slide"]["metrics"]["auroc"] is None
    assert report["slide"]["metrics"]["spearman_index_correlation"] is None
    assert all(np.isfinite(value) for value in metric_scalars(report).values())
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "scores,targets", [([], []), ([1], [0, 1]), ([float("nan")], [0]), ([0], [0.5])]
)
def test_rejects_invalid_inputs(scores, targets):
    with pytest.raises(ValueError):
        comparison_metrics(scores, targets, regression=False)


def collector(tmp_path, *, with_patient=True):
    mapping = pd.DataFrame({"path": ["s1.mrxs", "s2.mrxs", "s3.mrxs"]})
    if with_patient:
        mapping["record_num"] = ["001", "001", "002"]
    path = tmp_path / "mapping.csv"
    mapping.to_csv(path, index=False)
    return ComparisonTestCollector(
        {
            "label_mode": "index",
            "ml": {},
            "datamodule": {"test": {"data_mapping": str(path)}},
        }
    )


def test_patient_grouping_does_not_weight_patients_by_number_of_slides(tmp_path):
    c = collector(tmp_path)
    c.rows = [
        {"slide_id": "s1", "target": 0.2, "score": 0.9},
        {"slide_id": "s2", "target": 0.2, "score": -0.5},
        {"slide_id": "s3", "target": -0.2, "score": -0.4},
    ]
    report = c.report()
    assert report["patient_key"] == "record_num"
    assert report["slide"]["n"] == 3 and report["patient"]["n"] == 2
    assert report["patient"]["metrics"]["weighted_f1"] == 1
    assert report["repeated_evaluation"]["standard_deviation"] is None
    c.rows[1]["target"] = -0.2
    with pytest.raises(ValueError, match="Inconsistent targets"):
        c.report()


def test_never_infers_patient_identity_from_slide_id(tmp_path):
    c = collector(tmp_path, with_patient=False)
    c.rows = [{"slide_id": "s1", "target": 0.2, "score": 0.1}]
    assert c.report()["patient"] is None
    assert c.report()["patient_key"] is None


def test_collector_observes_exactly_existing_forward_calls_and_removes_hook(tmp_path):
    c = collector(tmp_path)

    class Model(torch.nn.Module):
        def forward(self, bag):
            return bag.mean().reshape(1), None

    model = Model()
    trainer = SimpleNamespace(world_size=1)
    rng = torch.get_rng_state().clone()
    c.on_test_start(trainer, model)
    batch = [(torch.tensor([[0.5]]), torch.tensor([0.2]), {"slide_id": "s1"})]
    c.on_test_batch_start(trainer, model, batch, 0)
    prediction, _ = model(batch[0][0])
    c.on_test_batch_end(trainer, model, None, batch, 0)
    c.on_test_end(trainer, model)
    assert c.rows[0]["score"] == prediction.item()
    assert len(model._forward_hooks) == 0
    assert torch.equal(rng, torch.get_rng_state())
    c.close()  # also safe in failure cleanup


def test_native_model_test_metrics_and_weights_are_unchanged(tmp_path):
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader

    from ml.data.datamodule import mil_collate
    from ml.models.aggregators import MeanPool
    from ml.models.encoders import IdentityEncoder
    from ml.models.heads import LinearHead
    from ml.models.module import MammaprintModule

    model = MammaprintModule(
        encoder=IdentityEncoder(out_dim=2),
        aggregator=MeanPool(feature_dim=2),
        head=LinearHead(in_dim=2),
        loss=torch.nn.MSELoss(),
    )
    with torch.no_grad():
        model.head.linear.weight.fill_(0.5)
        model.head.linear.bias.zero_()
    samples = [
        (torch.tensor([[0.4, 0.6]]), torch.tensor([0.2]), {"slide_id": "s1"}),
        (torch.tensor([[-0.4, -0.6]]), torch.tensor([-0.2]), {"slide_id": "s3"}),
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=mil_collate)
    c = collector(tmp_path)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    baseline = trainer.test(model, dataloaders=loader, verbose=False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    trainer.callbacks.append(c)
    observed = trainer.test(model, dataloaders=loader, verbose=False)
    assert observed == baseline
    assert [row["score"] for row in c.rows] == pytest.approx([0.5, -0.5])
    assert c.report()["patient"]["metrics"]["f1_a"] == 1
    assert not model._forward_hooks
    assert all(
        torch.equal(value, before[key]) for key, value in model.state_dict().items()
    )
