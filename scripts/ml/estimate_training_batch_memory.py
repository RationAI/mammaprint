"""Estimate per-fit CUDA memory for frozen-embedding MammaPrint MIL."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


GIB = 2**30
INPUT_WIDTH = 2560
ATTENTION_WIDTH = 512
MLP_WIDTH = 256
PROJECTOR_WIDTH = 256
PROJECTED_ATTENTION_WIDTH = 128
RUNTIME_GIB = 1.0
DEVICE_BUDGET_FRACTION = 0.8
BAG_SIZES = Path(__file__).with_name("training_batch_bag_sizes.csv")
PYRAMID_BAG_SIZES = Path(__file__).with_name("multiscale_batch_bag_sizes.csv")
L1_CAP = 3000


@dataclass(frozen=True)
class Profile:
    parameters: int
    train_floats_per_tile: int
    eval_floats_per_tile: int
    projected: bool = False


def layer_parameters(in_width: int, out_width: int) -> int:
    return (in_width + 1) * out_width


def attention_parameters(width: int, hidden: int) -> int:
    return 2 * layer_parameters(width, hidden) + layer_parameters(hidden, 1)


def profiles() -> dict[str, Profile]:
    width = INPUT_WIDTH
    hidden = ATTENTION_WIDTH
    projector = PROJECTOR_WIDTH
    gate = PROJECTED_ATTENTION_WIDTH
    projected_parameters = layer_parameters(width, projector) + layer_parameters(
        projector, 1
    )
    result = {
        "multiscale-mean-linear": Profile(
            7681, 2 * 21 * width + 3 * width, 2 * 21 * width + 3 * width
        ),
        "multiscale-projected-mean": Profile(
            671745,
            2 * 21 * width + 8 * 21 * projector + 3 * projector,
            2 * 21 * width + 3 * 21 * projector,
        ),
        "multiscale-slide-attention": Profile(
            1067266,
            2 * 21 * width + 8 * 21 * projector + 8 * 3 * projector + 8 * 256,
            2 * 21 * width + 3 * 21 * projector + 3 * 768 + 3 * 256,
        ),
        "abmil-mean": Profile(
            projected_parameters,
            width + 8 * projector + 16,
            width + 3 * projector + 8,
            True,
        ),
        "abmil-attention": Profile(
            projected_parameters + attention_parameters(projector, gate),
            width + 10 * projector + 8 * gate + 16,
            width + 4 * projector + 3 * gate + 8,
            True,
        ),
    }
    heads = {
        "linear": layer_parameters(width, 1),
        "mlp": layer_parameters(width, MLP_WIDTH) + layer_parameters(MLP_WIDTH, 1),
    }
    for head, parameters in heads.items():
        result[f"mean-{head}"] = result[f"max-{head}"] = Profile(
            parameters, 2 * width, 2 * width
        )
        result[f"attention-{head}"] = Profile(
            parameters + attention_parameters(width, hidden) + 2 * width,
            6 * width + 8 * hidden + 16,
            3 * width + 3 * hidden + 8,
        )
    return result


PROFILES = profiles()


@dataclass(frozen=True)
class BagSize:
    mean: float
    largest: int
    mean_capped_at_1024: float | None = None

    def capped(self, cap: int) -> "BagSize":
        mean = self.mean_capped_at_1024 if cap == 1024 else None
        return BagSize(
            min(self.mean, cap) if mean is None else mean, min(self.largest, cap)
        )


def bag_sizes(level: int, mask: str, *, multiscale: bool = False) -> dict[str, BagSize]:
    if multiscale and level != 4:
        raise ValueError("The recorded multiscale profiles use L4/L3/L2 pyramids")
    if level == 1 and mask == "cancer_mask_5":
        return dict.fromkeys(("train", "val", "test"), BagSize(L1_CAP, L1_CAP))
    path = PYRAMID_BAG_SIZES if multiscale else BAG_SIZES
    with path.open(newline="", encoding="utf-8") as file:
        sizes = {
            row["split"]: BagSize(
                float(row["mean_tiles"]),
                int(row["max_tiles"]),
                float(row["capped_1024_mean_tiles"]) if not multiscale else None,
            )
            for row in csv.DictReader(file)
            if int(row["level"]) == level and row["mask"] == mask
        }
    if set(sizes) != {"train", "val", "test"}:
        raise ValueError(f"No complete L{level} bag profile for mask {mask!r}")
    return sizes


def estimate(args: argparse.Namespace) -> dict[str, object]:
    profile = PROFILES[args.architecture]
    multiscale = args.architecture.startswith("multiscale-")
    sizes = bag_sizes(args.level, args.mask, multiscale=multiscale)
    if multiscale and args.train_tile_cap:
        raise ValueError("Multiscale recipes use uncapped retained regions")
    train_batch = args.train_batch_size or (4 if profile.projected else 1)
    eval_batch = args.eval_batch_size or (1 if profile.projected else train_batch)
    cap = (
        args.train_tile_cap
        if args.train_tile_cap is not None
        else (1024 if profile.projected else 0)
    )
    include_test = (
        args.include_test if args.include_test is not None else not profile.projected
    )
    if cap:
        sizes["train"] = sizes["train"].capped(cap)

    fixed_bytes = RUNTIME_GIB * GIB + 20 * profile.parameters
    stage_estimates = {}
    for split in ("train", "val", "test") if include_test else ("train", "val"):
        batch, floats_per_tile = (
            (train_batch, profile.train_floats_per_tile)
            if split == "train"
            else (eval_batch, profile.eval_floats_per_tile)
        )
        size = sizes[split]
        bytes_per_tile_batch = batch * 4 * floats_per_tile
        stage_estimates[split] = {
            "mean_bag_gib": (fixed_bytes + bytes_per_tile_batch * size.mean) / GIB,
            "largest_bag_gib": (fixed_bytes + bytes_per_tile_batch * size.largest)
            / GIB,
        }
    typical = max(stage["mean_bag_gib"] for stage in stage_estimates.values())
    largest = max(stage["largest_bag_gib"] for stage in stage_estimates.values())
    budget = math.ceil(largest * 4) / 4
    result: dict[str, object] = {
        "level": args.level,
        "mask": args.mask,
        "architecture": args.architecture,
        "bag_unit": "retained pyramid regions" if multiscale else "tiles",
        "bag_evidence": "recorded complete-pyramid audits"
        if multiscale
        else "selection-cap proxy"
        if args.level == 1
        else "recorded tile counts",
        "train_batch_size": train_batch,
        "eval_batch_size": eval_batch,
        "train_tile_cap": cap or None,
        "typical_gib_per_fit": round(typical, 3),
        "planning_gib_per_fit": budget,
        "stages": {
            split: {key: round(value, 3) for key, value in stage.items()}
            for split, stage in stage_estimates.items()
        },
    }
    if args.device_gib is not None:
        result["device_gib"] = args.device_gib
        result["vram_fit_count_ceiling"] = math.floor(
            args.device_gib * DEVICE_BUDGET_FRACTION / budget
        )
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=sorted(PROFILES))
    parser.add_argument("--level", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--mask", default="cancer_mask_5")
    parser.add_argument("--train-batch-size", type=positive_int)
    parser.add_argument("--eval-batch-size", type=positive_int)
    parser.add_argument(
        "--train-tile-cap", type=int, help="Use 0 for uncapped training"
    )
    parser.add_argument("--include-test", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--device-gib", type=float, help="Visible device capacity in GiB"
    )
    args = parser.parse_args(argv)
    if args.train_tile_cap is not None and args.train_tile_cap < 0:
        parser.error("--train-tile-cap must be nonnegative")
    if args.device_gib is not None and (
        not math.isfinite(args.device_gib) or args.device_gib <= 0
    ):
        parser.error("--device-gib must be finite and positive")
    try:
        print(json.dumps(estimate(args), indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
