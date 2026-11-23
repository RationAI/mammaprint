"""Summarize validation outcomes and build the HTML renderer's input bundle."""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any

import numpy as np

from ml.explainability.tile_removal_evaluate import BUDGETS


def exact_mask(row: dict[str, Any]) -> dict[str, Any] | None:
    return next((mask for mask in row["masks"] if mask["kind"] == "exact"), None)


def _distance(row: dict[str, Any], bands: list[float]) -> str:
    index = row["intact_abs_index"]
    if index is None:
        return "unavailable"
    index_band = int(np.searchsorted(bands, index, side="right"))
    if index_band == len(bands):
        return f"≥ {bands[-1]:g}"
    lower = bands[index_band - 1] if index_band else 0
    return f"[{lower:g}, {bands[index_band]:g})"


def _quantiles(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    q1, median, q3 = [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])]
    return {"count": len(values), "q1": q1, "median": median, "q3": q3}


def _budgets(records: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    return {
        budget: {
            "evaluable": len(records),
            "successful": sum(
                bool(mask["within_budgets"][budget]) for _, mask in records
            ),
            "unresolved": sum(
                mask["verification_status"]
                in {"minimum_unresolved", "boundary_unresolved"}
                and not mask["within_budgets"][budget]
                for _, mask in records
            ),
        }
        for budget in BUDGETS
    }


def summarize(rows: list[dict[str, Any]], bands: list[float]) -> dict[str, Any]:
    evaluable = []
    reasons: Counter[str] = Counter()
    strata: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    exact_minimum = []
    shifts = []
    extra_cost = []
    compact_attempted = compact_successful = 0
    controls = {
        family: {"requested": 0, "matched": 0, "evaluable": 0, "successful": 0}
        for family in ("exact", "compact")
    }
    for row in rows:
        compact = next(
            (item for item in row["masks"] if item["kind"] == "compact"), None
        )
        if compact is not None and compact["target_reached"] is not None:
            compact_attempted += 1
            compact_successful += bool(compact["target_reached"])
        for family, counts in row.get("controls", {}).items():
            controls[family]["requested"] += counts["requested"]
            controls[family]["matched"] += counts["matched"]
        for control in row["masks"]:
            if control["kind"] == "control" and control["target_reached"] is not None:
                counts = controls[control["family"]]
                counts["evaluable"] += 1
                counts["successful"] += bool(control["target_reached"])
        mask = exact_mask(row)
        if mask is None or mask["target_reached"] is None:
            reasons[row["status"] if mask is None else mask["verification_status"]] += 1
            continue
        evaluable.append((row, mask))
        consistency = "correct" if row["baseline_correct"] else "misclassified"
        direction = f"{row['intact_class']} to {row['target_class']}"
        strata[consistency, direction, _distance(row, bands)].append((row, mask))
        if not mask["target_reached"]:
            continue
        if mask["verification_status"] == "verified_minimum":
            exact_minimum.append(mask["deleted_fraction"])
        shifts.append(mask["masked_index"] - row["intact_index"])
        if (
            compact
            and compact["target_reached"]
            and mask["verification_status"] == "verified_minimum"
        ):
            extra_cost.append(compact["deleted_fraction"] - mask["deleted_fraction"])
    return {
        "total": len(rows),
        "evaluable": len(evaluable),
        "successful": sum(bool(mask["target_reached"]) for _, mask in evaluable),
        "unavailable_reasons": dict(reasons),
        "budgets": _budgets(evaluable),
        "minimum_deleted_fraction": _quantiles(exact_minimum),
        "successful_index_shift": _quantiles(shifts),
        "compact_extra_deleted_fraction": _quantiles(extra_cost),
        "compact": {"successful": compact_successful, "evaluable": compact_attempted},
        "controls": controls,
        "strata": [
            {
                "baseline": key[0],
                "direction": key[1],
                "distance_band": key[2],
                "budgets": _budgets(group),
            }
            for key, group in sorted(strata.items())
        ],
    }


def paired_summary(
    models: list[dict[str, Any]], bands: list[float]
) -> list[dict[str, Any]]:
    paired = []
    for first, second in combinations(models, 2):
        second_slides = {row["slide_id"]: row for row in second["slides"]}
        common = [
            (row, second_slides[row["slide_id"]])
            for row in first["slides"]
            if row["slide_id"] in second_slides
        ]
        matched = [
            (left, right)
            for left, right in common
            if left["target_class"] and left["target_class"] == right["target_class"]
        ]
        aligned = [
            (left, right)
            for left, right in matched
            if all(
                (mask := exact_mask(row)) is not None
                and mask["target_reached"] is not None
                for row in (left, right)
            )
        ]
        paired.append(
            {
                "model_ids": [first["model_id"], second["model_id"]],
                "common_slide_count": len(common),
                "aligned_target_count": len(matched),
                "jointly_evaluable_count": len(aligned),
                "models": [
                    summarize([pair[i] for pair in aligned], bands) for i in range(2)
                ],
            }
        )
    return paired


def choose_review(
    models: list[dict[str, Any]], options: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[
        tuple[str, str, str, bool, bool], deque[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(deque)

    def rank(*values: object) -> str:
        fields = "|".join(str(v) for v in (options["seed"], *values))
        return hashlib.sha256(fields.encode()).hexdigest()

    for model in models:
        candidates = sorted(model["slides"], key=lambda row: rank(row["slide_id"]))
        for row in candidates:
            exact = exact_mask(row)
            if exact is None or not exact["viewer_url"] or not exact["target_reached"]:
                continue
            compact = next(
                (mask for mask in row["masks"] if mask["kind"] == "compact"), None
            )
            group = (
                model["model_id"],
                row["target_class"],
                _distance(row, options["distance_bands"]),
                bool(compact and compact["target_reached"]),
                row["baseline_correct"],
            )
            groups[group].append((model, row))
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while groups and len(chosen) < options["review_slides"]:
        for key in sorted(groups):
            queue = groups[key]
            chosen.append(queue.popleft())
            if not queue:
                del groups[key]
            if len(chosen) == options["review_slides"]:
                break

    review = []
    key_rows = []
    for model, row in chosen:
        for mask in row["masks"]:
            if not mask["viewer_url"]:
                continue
            review_id = (
                "R"
                + hmac.new(
                    options["_review_key"],
                    f"review|{model['checkpoint_sha256']}|{row['slide_id']}|{mask['mask_id']}".encode(),
                    hashlib.sha256,
                ).hexdigest()[:15]
            )
            review.append({"review_id": review_id, "viewer_url": mask["viewer_url"]})
            key_rows.append(
                {
                    "review_id": review_id,
                    "model_id": model["model_id"],
                    "slide_id": row["slide_id"],
                    "patient_id": row["patient_id"],
                    "mask_id": mask["mask_id"],
                    "kind": mask["kind"],
                    "target_reached": mask["target_reached"],
                }
            )
    return sorted(review, key=lambda row: row["review_id"]), sorted(
        key_rows, key=lambda row: row["review_id"]
    )


def markdown_summary(models: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    lines = ["# Targeted tile removal on validation", ""]
    for model in models:
        summary = model["summary"]
        lines.extend(
            [
                f"## {model['model_id']}",
                "",
                f"Validation embeddings {model['validation_source']}",
                "",
                f"Verified target reached on {summary['successful']} of {summary['evaluable']} "
                f"evaluable slides. The roster contains {summary['total']} slides.",
                "",
                "| Budget | Successful | Evaluable | Unresolved within budget |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, budget in summary["budgets"].items():
            lines.append(
                f"| {label} | {budget['successful']} | {budget['evaluable']} | {budget['unresolved']} |"
            )
        lines.append("")
        if summary["unavailable_reasons"]:
            reasons = ", ".join(
                f"{key} ({value})"
                for key, value in summary["unavailable_reasons"].items()
            )
            lines.extend([f"Unavailable cases {reasons}", ""])
        compact = summary["compact"]
        lines.extend(
            [
                f"Compact masks reached the target on {compact['successful']} of "
                f"{compact['evaluable']} evaluable slides at the 10% cap.",
                "",
                "| Control family | Requested | Matched | Evaluable | Flipped |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for family, control in summary["controls"].items():
            lines.append(
                f"| {family} | {control['requested']} | {control['matched']} | "
                f"{control['evaluable']} | {control['successful']} |"
            )
        lines.append("")
        for label, field in (
            ("Minimum deleted fraction", "minimum_deleted_fraction"),
            ("Index shift after successful flips", "successful_index_shift"),
            (
                "Additional fraction removed by compact masks",
                "compact_extra_deleted_fraction",
            ),
        ):
            if stats := summary[field]:
                lines.append(
                    f"{label} had a median of {stats['median']:.4g} across {stats['count']} "
                    f"slides (IQR {stats['q1']:.4g} to {stats['q3']:.4g})."
                )
        lines.extend(
            [
                "",
                "| Baseline | Direction | Absolute intact index | Evaluable | 1% | 5% | 10% |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in summary["strata"]:
            budgets = item["budgets"]
            lines.append(
                f"| {item['baseline']} | {item['direction']} | {item['distance_band']} | "
                f"{budgets['1%']['evaluable']} | {budgets['1%']['successful']} | "
                f"{budgets['5%']['successful']} | {budgets['10%']['successful']} |"
            )
        lines.append("")
    if paired:
        lines.extend(["## Common validation slides", ""])
    for pair in paired:
        first, second = pair["model_ids"]
        lines.append(
            f"{first} and {second} shared {pair['common_slide_count']} slides. "
            f"{pair['jointly_evaluable_count']} were jointly evaluable with the same target "
            f"direction, out of {pair['aligned_target_count']} with aligned targets."
        )
        lines.extend(
            ["", "| Budget | " + first + " | " + second + " |", "| --- | ---: | ---: |"]
        )
        for budget in BUDGETS:
            counts = [
                model["budgets"][budget]["successful"] for model in pair["models"]
            ]
            lines.append(f"| {budget} | {counts[0]} | {counts[1]} |")
        lines.append("")
    return "\n".join(lines)
