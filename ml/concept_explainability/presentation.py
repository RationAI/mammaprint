"""Human-facing signs, independent of the retained upstream evaluation schema."""

import re

import numpy as np
import pandas as pd


EFFECT_DISPLAY_VERSION = "after-minus-before-v1"
EFFECT_NOTE = f"""<!-- concept-effect-display:{EFFECT_DISPLAY_VERSION} -->
<p>Both change columns mean <b>prediction after change minus prediction before change</b>.
Positive values increase the MammaPrint index or P(A), toward A; negative values move toward B.
Classification always shows changes in P(A), including for the predicted-B cohort, in percentage points.
Regression changes are in raw index units. Increase concept and decrease concept are separate
interventions using the configured standardized step in across-slide mean concept share, scaled by each tile's total coefficient amount;
decreases stop at zero. Changes are measured against reconstructed-model output, not the original embeddings.
Direction consistency is the percentage of slides whose two-sided summary points in the cohort-average direction;
it is not accuracy. Two-sided agreement checks whether increasing and decreasing the concept support the same direction.
Zero-limited tiles is the mean percentage of tiles where the requested decrease would cross zero.
Presence is a group-to-cohort ratio; for all slides it is 1 by definition, not 100% prevalence.
These are model responses, not biological causal effects.</p>"""


def effect_display_frame(effects: pd.DataFrame) -> pd.DataFrame:
    """Convert upstream signs to actual output deltas without mutating raw data."""
    classification = effects.target == "luminal_a_probability"
    valid = (
        classification
        & effects.group.isin(["A", "B"])
        & (effects.units == "percentage_points")
    ) | (
        (effects.target == "mammaprint_index")
        & (effects.group == "all")
        & (effects.units == "index")
    )
    if not valid.all():
        raise ValueError("Unknown target/group/units in concept effects.")
    if not np.isfinite(effects[["add", "subtract"]].to_numpy()).all():
        raise ValueError("Nonfinite concept effects.")
    # Stored B-cohort effects use P(B); stored subtraction is before - after.
    factor = np.where(classification & (effects.group == "B"), -1, 1)
    return pd.DataFrame(
        {
            "Output": effects.target.map(
                {
                    "mammaprint_index": "MammaPrint index",
                    "luminal_a_probability": "P(A)",
                }
            ),
            "Cohort": effects.group.map(
                {
                    "all": "All slides",
                    "A": "Predicted A slides",
                    "B": "Predicted B slides",
                }
            ),
            "Slides": effects.n_slides,
            "Increase concept: change": factor * effects["add"],
            "Decrease concept: change": -factor * effects["subtract"],
            "Units": effects.units.map(
                {"index": "index units", "percentage_points": "percentage points"}
            ),
            "Direction consistency (%)": 100 * effects.same_direction,
            "Two-sided agreement": effects.disagrees.map({False: "Yes", True: "No"}),
            "Zero-limited tiles (%)": 100 * effects.clipped_tiles,
            "Status": effects.status,
        }
    )


def render_effect_table(effects: pd.DataFrame) -> str:
    frame = effect_display_frame(effects)
    return frame.to_html(
        index=False,
        escape=True,
        formatters={
            "Increase concept: change": "{:+.6f}".format,
            "Decrease concept: change": "{:+.6f}".format,
            "Direction consistency (%)": "{:.1f}".format,
            "Zero-limited tiles (%)": "{:.1f}".format,
        },
    )


def upgrade_effect_html(page: str, effects: pd.DataFrame) -> str:
    """Replace only verified effect tables and interpretation text in old HTML.

    Exact old-table matching deliberately rejects unfamiliar templates rather
    than risking modification of labels, contributions, images or viewer links.
    Reapplying to an already-upgraded matching report is a no-op.
    """
    if effects.empty or effects.duplicated(["target", "group", "concept"]).any():
        raise ValueError("Missing or duplicate concept effects.")
    marker = f"<!-- concept-effect-display:{EFFECT_DISPLAY_VERSION} -->"
    updated = marker in page
    sections = re.findall(r"<section><h2>Concept (\d+)</h2>", page)
    if sorted(map(int, sections)) != sorted(effects.concept.unique()):
        raise ValueError("HTML concepts do not match the supplied effect table.")
    section_pattern = r"<section><h2>Concept (\d+)</h2>.*?</section>"
    if len(re.findall(section_pattern, page, flags=re.DOTALL)) != len(sections):
        raise ValueError("Unrecognized concept section structure.")

    def replace_section(match):
        rows = effects[effects.concept == int(match[1])]
        new = render_effect_table(rows)
        old = new if updated else rows.to_html(index=False, escape=True)
        section = match[0]
        if section.count(old) != 1:
            raise ValueError("HTML effect table does not match the supplied raw data.")
        return section.replace(old, new, 1)

    page = re.sub(section_pattern, replace_section, page, flags=re.DOTALL)
    if updated:
        if page.count(EFFECT_NOTE) != 1:
            raise ValueError("Unrecognized updated interpretation text.")
        return page
    pattern = r"<p>Effects compare two-sided concept changes against reconstructed-model output, not the original embeddings\..*?</p>"
    if len(re.findall(pattern, page, flags=re.DOTALL)) != 1:
        raise ValueError("Unrecognized legacy interpretation text.")
    return re.sub(pattern, lambda _: EFFECT_NOTE, page, count=1, flags=re.DOTALL)
