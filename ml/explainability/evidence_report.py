"""Self-contained HTML evidence section, added to the standard RationAI report."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_evidence(path: Path) -> str:
    data = json.loads(path.read_text())
    # JSON exports must remain usable by strict parsers, with finite numbers only.
    json.dumps(data, allow_nan=False)
    for slide in data["slides"]:
        url = slide.get("viewer_url")
        if url and urlsplit(url).scheme not in {"https", "http"}:
            slide["viewer_url"] = None
        for target in slide["targets"].values():
            histogram = target["histogram"]
            maximum = max(histogram["counts"], default=1) or 1
            histogram["heights"] = [
                80 * count / maximum for count in histogram["counts"]
            ]
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(("html",)),
    )
    environment.filters["number"] = lambda value: f"{value:+.6g}"
    return environment.get_template("mean_evidence.html").render(data=data)


def evidence_document(section: str) -> str:
    """Fallback when numerical evidence exists but no WSI masks can be shown."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>MammaPrint tile evidence</title></head><body>"
        "<h1>MammaPrint tile explanations</h1>"
        "<p>No rendered slide masks are available. Numerical evidence is shown below.</p>"
        + section
        + "</body></html>"
    )
