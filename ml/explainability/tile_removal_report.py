"""Shared HTML rendering and provenance checks for targeted tile removal."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping
    from pathlib import Path


_BEGIN = "BEGIN MAMMAPRINT TARGETED TILE REMOVAL"
_END = "END MAMMAPRINT TARGETED TILE REMOVAL"
_RATIONAI_HEADING = "MammaPrint tile explanations"
_STANDALONE_TITLES = {
    "MammaPrint tile-explainability review",
    "MIL tile-explanation review",
    "MammaPrint targeted tile removal",
}
_BUDGETS = ("1%", "5%", "10%")
_MASK_NAMES = {
    "exact": "Scattered mask",
    "class": "Scattered mask",
    "primary": "Scattered mask",
    "compact": "Compact mask",
    "alternative": "Alternative mask",
    "control": "Matched control",
    "index": "Index-band mask",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(data: Mapping[str, Any]) -> tuple[str, str]:
    run = data.get("source_run_id")
    digest = data.get("checkpoint_sha256")
    if not isinstance(run, str) or not re.fullmatch(r"[a-f0-9]{32}", run):
        raise ValueError("Missing or invalid source training run ID.")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Missing or invalid checkpoint SHA-256.")
    return run, digest


def _slides(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in model["slides"]:
        slide_id = row.get("slide_id")
        if not isinstance(slide_id, str) or not slide_id.strip() or slide_id in result:
            raise ValueError(
                "Tile-removal results contain an invalid or repeated slide ID."
            )
        result[slide_id] = row
    if not result:
        raise ValueError("Tile-removal results contain no validation slides.")
    return result


@dataclass(frozen=True)
class ReportModel:
    model: dict[str, Any]
    split: str
    bundle_sha256: str


def load_report_model(
    bundle: Path,
    provenance: Mapping[str, Any],
    *,
    explanation_run_id: str | None = None,
) -> ReportModel:
    split = provenance.get("split")
    if split not in {"val", "test"}:
        raise ValueError("Expected a validation or test tile-explanation report.")
    expected = _identity(provenance)
    if explanation_run_id is not None and (
        provenance.get("explanation_run_id") != explanation_run_id
        or expected[0] == explanation_run_id
    ):
        raise ValueError("The manifest does not identify this explanation run.")
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("split") != "val" or manifest.get("pilot_slide_ids"):
        raise ValueError("Expected a full validation tile-removal bundle.")
    data_bytes = (bundle / "report_data.json").read_bytes()
    if manifest.get("report_data_sha256") != sha256(data_bytes):
        raise ValueError("The report data hash does not match the validation manifest.")
    manifest_matches = [row for row in manifest["models"] if _identity(row) == expected]
    matches = [
        row for row in json.loads(data_bytes)["models"] if _identity(row) == expected
    ]
    if (
        len(manifest_matches) != 1
        or len(matches) != 1
        or manifest_matches[0]["model_id"] != matches[0]["model_id"]
    ):
        raise ValueError("The report checkpoint has no unique match in the bundle.")
    _slides(matches[0])
    return ReportModel(matches[0], split, sha256(manifest_bytes))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _label(value: Any) -> str:
    return (
        _escape(str(value).replace("_", " ").capitalize()) if value else "Unavailable"
    )


def _number(value: Any, *, percent: bool = False) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Unavailable"
    return format(value, ".1%" if percent else ".4g")


def _url(value: Any) -> str | None:
    if not isinstance(value, str) or any(
        char.isspace() or ord(char) < 32 or char == "\\" for char in value
    ):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
    ):
        return None
    return value


def missing_overlay_count(model: Mapping[str, Any]) -> int:
    return sum(
        bool(mask.get("overlay_expected", mask.get("overlay_path")))
        and _url(mask.get("viewer_url")) is None
        for slide in _slides(model).values()
        for mask in slide.get("masks", ())
    )


def _exact(slide: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        (
            mask
            for mask in slide.get("masks", ())
            if mask.get("kind") in {"exact", "class", "primary"}
        ),
        {},
    )


def _summary(model: Mapping[str, Any], slides: Mapping[str, Any]) -> str:
    summary = model.get("summary", {})
    budgets = summary.get("budgets", {})
    eligible = [
        slide
        for slide in slides.values()
        if slide.get("intact_index") is not None
        and slide.get("target_class") is not None
    ]
    values = []
    for label in _BUDGETS:
        recorded = budgets.get(label)
        if isinstance(recorded, dict):
            hits = recorded["successful"]
            denominator = recorded["evaluable"]
        else:
            denominator = len(eligible)
            cap = int(label.rstrip("%")) / 100
            hits = (
                recorded
                if isinstance(recorded, int)
                else sum(
                    bool(mask.get("target_reached"))
                    and isinstance(mask.get("deleted_fraction"), (int, float))
                    and mask["deleted_fraction"] <= cap
                    for mask in map(_exact, eligible)
                )
            )
        values.append(
            "<tr>"
            f'<th scope="row">{label}</th><td>{int(hits)}</td><td>{int(denominator)}</td>'
            "</tr>"
        )
    compact = summary.get("compact")
    comparison = ""
    if compact:
        comparison = (
            f"<p>Compact masks flipped {int(compact['successful'])} of "
            f"{int(compact['evaluable'])} evaluable slides at the 10% cap.</p>"
        )
    for family, counts in summary.get("controls", {}).items():
        comparison += (
            f"<p>{_label(family)} controls matched {int(counts['matched'])} of "
            f"{int(counts['requested'])} requested regions. "
            f"{int(counts['successful'])} of {int(counts['evaluable'])} evaluable controls flipped.</p>"
        )
    return (
        f"<p>{len(slides)} validation slides in the result bundle.</p>"
        "<table><caption>Verified class flips by deletion budget</caption>"
        '<thead><tr><th scope="col">Budget</th><th scope="col">Flipped</th>'
        '<th scope="col">Evaluable</th></tr></thead><tbody>'
        + "".join(values)
        + "</tbody></table>"
        + comparison
    )


def _mask_row(mask: Mapping[str, Any], coverage_fields: dict[str, str]) -> str:
    kind = mask.get("kind")
    name = _MASK_NAMES.get(str(kind)) or _label(kind)
    if kind in {"exact", "class", "primary"} and mask.get("verification_status") == (
        "verified_minimum"
    ):
        name = "Minimum tile mask"
    if kind == "control" and mask.get("family") in {"exact", "compact"}:
        name = f"Matched {_escape(mask['family'])} control"
    if kind == "index" and (band := mask.get("index_band")) and len(band) == 2:
        name = f"{name} [{_number(band[0])}, {_number(band[1])}]"
    viewer_url = _url(mask.get("viewer_url"))
    link = (
        f'<a href="{_escape(viewer_url)}" target="_blank" rel="noopener noreferrer">'
        "Open in xOpat</a>"
        if viewer_url
        else "Unavailable"
    )
    coverage = "".join(
        f"<td>{_number(mask.get(key), percent=True)}</td>" for key in coverage_fields
    )
    return (
        f'<tr><th scope="row">{name}</th>'
        f"<td>{_number(mask.get('masked_index'))}</td>"
        f"<td>{_escape(mask.get('masked_class') or 'Unavailable')}</td>"
        f"<td>{_escape(mask.get('deleted_count', 'Unavailable'))}</td>"
        f"<td>{_number(mask.get('deleted_fraction'), percent=True)}</td>"
        f"<td>{_label(mask.get('verification_status'))}</td>{coverage}<td>{link}</td></tr>"
    )


def _slide_panel(slide_id: str, slide: Mapping[str, Any]) -> str:
    panel = (
        f'<details data-tile-removal-slide-id="{_escape(slide_id)}">'
        f"<summary>{_escape(slide_id)} · {_label(slide.get('status'))}</summary>"
    )
    intact = slide.get("intact_index")
    if intact is not None:
        panel += (
            f"<p>Intact index {_number(intact)} · Class "
            f"{_escape(slide.get('intact_class') or 'Unavailable')} · Target "
            f"{_escape(slide.get('target_class') or 'Unavailable')}</p>"
        )
    if slide.get("masks"):
        coverage_fields = {
            key: label
            for key, label in (
                ("retained_tumor_fraction", "Tumor retained"),
                ("retained_attention_mass", "Original attention retained"),
            )
            if any(mask.get(key) is not None for mask in slide["masks"])
        }
        coverage_headings = "".join(
            f'<th scope="col">{label}</th>' for label in coverage_fields.values()
        )
        panel += (
            '<div class="tile-removal-table-scroll"><table><thead><tr>'
            '<th scope="col">Mask</th><th scope="col">Masked index</th>'
            '<th scope="col">Masked class</th><th scope="col">Tiles removed</th>'
            '<th scope="col">Fraction removed</th><th scope="col">Verification</th>'
            + coverage_headings
            + '<th scope="col">Overlay</th></tr></thead><tbody>'
            + "".join(_mask_row(mask, coverage_fields) for mask in slide["masks"])
            + "</tbody></table></div>"
        )
    return panel + "</details>"


def render_tile_removal_section(
    model: Mapping[str, Any],
    *,
    report_split: str,
    slide_ids: Collection[str] | None = None,
    bundle_sha256: str | None = None,
) -> str:
    if report_split not in {"val", "test"}:
        raise ValueError("Expected a validation or test tile-explanation report.")
    slides = _slides(model)
    requested = sorted(slide_ids if report_split == "val" and slide_ids else slides)
    heading = "Targeted tile removal"
    if report_split == "test":
        heading += " on validation"
    panels = "".join(
        _slide_panel(
            slide_id,
            slides.get(slide_id, {"status": "validation result unavailable"}),
        )
        for slide_id in requested
    )
    findings = model.get("review_findings")
    findings_html = (
        '<h3>Pathologist review findings</h3><p class="tile-removal-findings">'
        f"{_escape(findings)}</p>"
        if findings
        else ""
    )
    run, digest = _identity(model)
    bundle_attribute = (
        f' data-bundle-sha256="{_escape(bundle_sha256)}"' if bundle_sha256 else ""
    )
    return (
        "\n"
        f"<!-- {_BEGIN} -->\n"
        '<section id="mammaprint-targeted-tile-removal" '
        'aria-labelledby="mammaprint-targeted-tile-removal-heading" '
        f'data-source-run-id="{run}" data-checkpoint-sha256="{digest}"{bundle_attribute}>'
        "<style>#mammaprint-targeted-tile-removal{margin:2rem auto;"
        "max-width:90rem;padding:1rem}#mammaprint-targeted-tile-removal table"
        "{border-collapse:collapse}#mammaprint-targeted-tile-removal td,"
        "#mammaprint-targeted-tile-removal th{padding:.35rem;text-align:left;"
        "border-bottom:1px solid currentColor}"
        "#mammaprint-targeted-tile-removal .tile-removal-table-scroll"
        "{overflow:auto}#mammaprint-targeted-tile-removal details"
        "{margin:.5rem 0}#mammaprint-targeted-tile-removal .tile-removal-findings"
        "{white-space:pre-wrap}</style>"
        f'<h2 id="mammaprint-targeted-tile-removal-heading">{heading}</h2>'
        f"<p>{_escape(model['model_id'])} · Validation</p>"
        + _summary(model, slides)
        + panels
        + findings_html
        + "</section>\n"
        f"<!-- {_END} -->\n"
    )


class _ReportParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_offsets = [0, *(m.end() for m in re.finditer("\n", source))]
        self.title = ""
        self.in_title = False
        self.headings: list[str] = []
        self.in_heading = False
        self.body_ends: list[int] = []
        self.begin: list[int] = []
        self.end: list[int] = []
        self.slide_ids: list[str] = []
        self.feed(source)
        self.close()

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_title |= tag == "title"
        if tag == "h1":
            self.headings.append("")
            self.in_heading = True
        fields = dict(attrs)
        if tag == "article" and (slide_id := fields.get("data-slide-id")):
            self.slide_ids.append(slide_id)
        if tag == "details" and (slide_id := fields.get("data-tile-removal-slide-id")):
            self.slide_ids.append(slide_id)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "h1":
            self.in_heading = False
        if tag == "body":
            self.body_ends.append(self._offset())

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.in_heading:
            self.headings[-1] += data

    def handle_comment(self, data: str) -> None:
        if data.strip() == _BEGIN:
            self.begin.append(self._offset())
        if data.strip() == _END:
            close = self.source.index("-->", self._offset()) + 3
            self.end.append(close)


def update_report_html(
    original: bytes,
    model: ReportModel,
    *,
    slide_ids: Collection[str] | None = None,
    explanation_run_id: str | None = None,
) -> bytes:
    source = original.decode("utf-8")
    parsed = _ReportParser(source)
    title = parsed.title.strip()
    rationai = title == "Test Report" and [
        heading.strip() for heading in parsed.headings
    ] == [_RATIONAI_HEADING]
    if (not rationai and title not in _STANDALONE_TITLES) or len(parsed.body_ends) != 1:
        raise ValueError("Unsupported report template or ambiguous HTML anchor.")
    if explanation_run_id is not None and (
        not re.fullmatch(r"[a-f0-9]{32}", explanation_run_id)
        or explanation_run_id not in source
    ):
        raise ValueError(
            "The report HTML does not identify the selected explanation run."
        )
    if model.split == "val":
        slide_ids = slide_ids if slide_ids is not None else set(parsed.slide_ids)
        if not slide_ids:
            raise ValueError("The validation report slide roster is missing or empty.")
    fragment = render_tile_removal_section(
        model.model,
        report_split=model.split,
        slide_ids=slide_ids,
        bundle_sha256=model.bundle_sha256,
    )
    if not (parsed.begin or parsed.end):
        start = end = parsed.body_ends[0]
    elif (
        len(parsed.begin) == len(parsed.end) == 1
        and parsed.begin[0] < parsed.end[0] <= parsed.body_ends[0]
    ):
        start, end = parsed.begin[0], parsed.end[0]
        # New sections include their surrounding newlines in the splice.
        if source[start - 1 : start] == "\n":
            start -= 1
        if source[end : end + 1] == "\n":
            end += 1
    else:
        raise ValueError("Ambiguous targeted tile-removal section markers.")
    return (source[:start] + fragment + source[end:]).encode("utf-8")


def render_standalone_report(model: Mapping[str, Any]) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>MammaPrint targeted tile removal</title></head><body>"
        "<h1>MammaPrint validation report</h1>"
        + render_tile_removal_section(model, report_split="val")
        + "</body></html>\n"
    )


def render_blinded_review(items: Iterable[Mapping[str, Any]]) -> str:
    cards = []
    seen = set()
    for item in items:
        review_id = item.get("review_id")
        if (
            not isinstance(review_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", review_id)
            or review_id in seen
        ):
            raise ValueError("Review IDs must be unique opaque identifiers.")
        seen.add(review_id)
        viewer_url = _url(item.get("viewer_url"))
        link = (
            f'<p><a href="{_escape(viewer_url)}" target="_blank" '
            'rel="noopener noreferrer">Open region in xOpat</a></p>'
            if viewer_url
            else "<p>Viewer unavailable.</p>"
        )
        cards.append(
            f'<fieldset data-review-id="{review_id}"><legend>{review_id}</legend>{link}'
            '<label>Morphology <textarea name="morphology"></textarea></label>'
            "<label>Tumor, stroma, and epithelium mix "
            '<textarea name="tissue_mix"></textarea></label>'
            '<label>Artifacts <textarea name="artifacts"></textarea></label>'
            '<label>Confidence <select name="confidence"><option value=""></option>'
            "<option>Low</option><option>Medium</option><option>High</option>"
            "</select></label></fieldset>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Blinded region review</title>"
        "<style>body{max-width:56rem;margin:2rem auto;padding:0 1rem}"
        "fieldset{margin:1rem 0}label{display:block;margin:.5rem}"
        "textarea{display:block;width:95%}</style></head><body>"
        "<h1>Blinded region review</h1>"
        + "".join(cards)
        + """<button type="button" id="download-review">Download responses</button>
<script>
document.getElementById("download-review").addEventListener("click", () => {
    const responses = Array.from(document.querySelectorAll("[data-review-id]"), row => ({
        review_id: row.dataset.reviewId,
        ...Object.fromEntries(Array.from(row.querySelectorAll("[name]"),
            field => [field.name, field.value])),
    }));
    const file = new Blob([JSON.stringify({responses}, null, 2)], {type: "application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(file);
    link.download = "blinded-review-responses.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 500);
});
</script></body></html>
"""
    )
