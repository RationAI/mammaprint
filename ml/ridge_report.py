"""Write the development and held-out screening reports."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from ml.ridge_screening import BASELINE, METRICS


if TYPE_CHECKING:
    from pathlib import Path


def estimate(value: float | None, se: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.5f}" if se is None else f"{value:.5f} ± {se:.5f}"


def write_report(
    output: Path,
    run: dict[str, Any],
    sanity: dict[str, Any],
    ranking: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> None:
    result = sanity["ridge"]["historical"]
    history = sanity["historical"]
    lines = [
        "# Ridge slide-statistics screening",
        "",
        "## Historical validation sanity check",
        "",
        f"L2 cm5 ({sanity['transform']}) ridge F1 was {result['f1']:.2%} "
        f"and MAE was {result['mae']:.5f}. Historical F1 ranged from {history['f1_min']:.2%} "
        f"to {history['f1_max']:.2%}, with mean MAE {history['mae_mean']:.5f}. "
        f"Checkpoint parity maximum error was {sanity['parity']['max_abs_error']:.3g}.",
        "",
    ]
    if sanity["record_overlap"]:
        lines.extend(
            [
                f"The historical split shared {len(sanity['record_overlap'])} record(s) across train and validation.",
                "",
            ]
        )
    if sanity["resolution"]:
        lines.extend([f"Sanity decision — {sanity['resolution']}", ""])
    lines.extend(["## Development ranking", ""])
    if ranking:
        lines.extend(
            [
                f"The shared ranking cohort has {ranking[0]['slides']} slides and {ranking[0]['records']} records. "
                "Errors are overlap-adjusted screening SEs. Deltas are candidate minus LayerNormed L2 cm5.",
                "",
                "| Feature set | MAE | Δ MAE | Spearman | Δ Spearman | Sign-AUROC | Δ Sign-AUROC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        delta = {
            (row["feature_set"], row["metric"]): estimate(row["delta"], row["se"])
            for row in contrasts
            if row["screen"] == run["ranking_scope"]
        }
        for row in ranking:
            arm = row["feature_set"]
            cells = [arm]
            for metric in METRICS:
                cells.extend(
                    [
                        estimate(row[metric], row[f"{metric}_se"]),
                        delta.get((arm, metric), "Reference"),
                    ]
                )
            lines.append("| " + " | ".join(cells) + " |")
        winner = run["winner"]
        blocks = ", ".join(
            sorted({source for source, _ in run["feature_blocks"][winner]})
        )
        lines.extend(
            [
                "",
                f"Recommended ridge baseline is **{winner}**. Carry **{blocks}** into the option 4 comparison "
                "and compare on the same eligible records and evaluation splits.",
                "",
            ]
        )
    else:
        lines.extend(["A common-cohort ranking is unavailable for this run.", ""])

    curve = [
        row
        for row in summaries
        if row["screen"] == "learning-curve" and row["feature_set"] == BASELINE
    ]
    if curve:
        lines.extend(
            ["## Learning curve", "", "![Learning curve](learning_curve.png)", ""]
        )
        for row in curve:
            lines.append(
                f"- {row['fraction']:.0%} used {row['fit_records_min']} to {row['fit_records_max']} training "
                f"records per fold. MAE {estimate(row['mae'], row['mae_se'])}."
            )
        end = next(
            row
            for row in contrasts
            if row["screen"] == "learning-curve"
            and row["feature_set"] == BASELINE
            and row["metric"] == "mae"
            and row["fraction"] == 1
        )
        lines.extend(
            ["", f"The 75% to 100% MAE change was {estimate(end['delta'], end['se'])}."]
        )
        if end["se"] is not None and end["delta"] + 2 * end["se"] < 0:
            lines.append(
                "The last increase in training records improved MAE and suggests that more data may help."
            )
        else:
            lines.append(
                "The last increase in training records gave no clear evidence of further improvement."
            )
        write_curve(output, curve)

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Available feature set | Development slides | Records |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in coverage:
        lines.append(f"| {row['feature_set']} | {row['slides']} | {row['records']} |")
    if run["missing_development_targets"]:
        lines.extend(
            [
                "",
                f"{run['missing_development_targets']} development slide(s) had missing or nonfinite targets.",
            ]
        )
    for item, reason in (*run["skipped"].items(), *run["errors"].items()):
        lines.extend(["", f"- {item} — {reason}"])
    lines.extend(
        [
            "",
            f"Cache {run['cache_path']} ({run['cache_digest']}).",
            "",
            "[Fold ledger](folds.csv) · [Fold metrics](metrics.csv) · [Summaries](summary.csv) · "
            "[Paired contrasts](contrasts.csv) · [Exclusions](exclusions.csv)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def write_curve(output: Path, curve: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 3.75), layout="constrained")
    x = [100 * row["fraction"] for row in curve]
    y = [row["mae"] for row in curve]
    error = [
        row["mae_se"] if row["mae_se"] is not None else float("nan") for row in curve
    ]
    axis.errorbar(x, y, yerr=error, color="#234a75", marker="o", capsize=4)
    axis.set_xticks(x, [f"{value:g}%" for value in x])
    axis.set(
        xlabel="Outer training records used",
        ylabel="Mean outer MAE",
        title="L2 cm5 ridge learning curve",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "learning_curve.png", dpi=150)
    plt.close(fig)


def test_section(result: dict[str, Any]) -> str:
    text = [
        "## Held-out test",
        "",
        f"Frozen **{result['feature']}** scored {result['scored_slides']} of "
        f"{result['expected_slides']} test slides across {result['scored_records']} records.",
        "",
    ]
    for name in METRICS:
        text.append(
            f"- {name.replace('_', ' ').upper()} {estimate(result[name], None)}"
        )
    exclusions = Counter(result["exclusions"].values())
    for reason, count in sorted(exclusions.items()):
        text.append(f"- {count} excluded slide(s) — {reason}")
    text.extend(["", f"Frozen model digest {result['lock_digest']}.", ""])
    return "\n".join(text)
