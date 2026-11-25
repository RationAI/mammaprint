import csv
import json

import pytest

from scripts.ml import estimate_training_batch_memory as memory


def read_estimate(capsys, level, architecture, *options):
    memory.main(["--level", str(level), "--architecture", architecture, *options])
    return json.loads(capsys.readouterr().out)


def test_reference_inventory_covers_all_masks_and_levels():
    with memory.BAG_SIZES.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    profiles = {(row["level"], row["mask"], row["split"]) for row in rows}
    masks = {
        "cancer_mask_5",
        "cancer_mask_9",
        "tissue_only",
        "epithel",
        "epithel_and_cancer_5",
        "epithel_or_cancer_5",
    }
    assert len(rows) == len(profiles)
    assert profiles == {
        (str(level), mask, split)
        for level in range(2, 6)
        for mask in masks
        for split in ("train", "val", "test")
    }
    for row in rows:
        assert (
            0
            < float(row["capped_1024_mean_tiles"])
            <= min(1024, float(row["mean_tiles"]))
            <= int(row["max_tiles"])
        )
        assert 0 < int(row["p95_tiles"]) <= int(row["max_tiles"])


def test_l2_attention_ceiling_uses_budget_and_headroom(capsys):
    estimate = read_estimate(capsys, 2, "attention-linear", "--device-gib", "9.50")
    assert estimate["typical_gib_per_fit"] == 1.63
    assert estimate["planning_gib_per_fit"] == 4.0
    assert estimate["vram_fit_count_ceiling"] == 1
    assert set(estimate["stages"]) == {"train", "val", "test"}


def test_abmil_cap_leaves_evaluation_bags_unchanged(capsys):
    default = read_estimate(capsys, 2, "abmil-mean")
    uncapped = read_estimate(
        capsys, 2, "abmil-mean", "--train-tile-cap", "0", "--include-test"
    )
    assert default["train_batch_size"] == 4 and default["eval_batch_size"] == 1
    assert default["train_tile_cap"] == 1024
    assert default["planning_gib_per_fit"] == 1.5
    assert set(default["stages"]) == {"train", "val"}
    assert default["stages"]["val"] == uncapped["stages"]["val"]
    assert (
        default["stages"]["train"]["largest_bag_gib"]
        < uncapped["stages"]["train"]["largest_bag_gib"]
    )


def test_l1_uses_a_cap_proxy_and_rejects_unknown_masks(capsys):
    estimate = read_estimate(capsys, 1, "mean-linear")
    assert estimate["bag_evidence"] == "selection-cap proxy"
    assert (
        estimate["stages"]["val"]["mean_bag_gib"]
        == estimate["stages"]["val"]["largest_bag_gib"]
    )
    with pytest.raises(SystemExit) as result:
        read_estimate(capsys, 1, "mean-linear", "--mask", "tissue_only")
    assert result.value.code == 2
