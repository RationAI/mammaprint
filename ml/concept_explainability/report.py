"""Flat, blinded concept naming report adapted from Adam Kukučka's IBD report.

The exact deployed naming_k16.html generator was not available. The layout,
coefficient-share selection, presence overlays and intervention descriptions
follow the pinned IBD source; the small response form follows the supplied brief.
"""

import base64
import html
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.concept_explainability.data import iter_slides, source_dataset
from ml.concept_explainability.evaluation import (
    dictionary_n_components,
    load_dictionary,
    read_codes,
    save_json,
    validate_n_components,
)
from ml.concept_explainability.upstream import code_share


FORM_SCRIPT = r"""
let revealed = false;
function revealResults() {
  document.querySelectorAll('.expected').forEach(field => field.disabled = true);
  document.body.classList.remove('blind');
  document.getElementById('reveal').disabled = true;
  revealed = true;
}
async function copyAnswers() {
  const answers = Array.from({length: REPORT_ID.n_components}, (_, concept) => ({
    concept,
    name: document.getElementById('name-'+concept).value,
    unclear_or_mixed: document.getElementById('unclear-'+concept).checked,
    expected: document.getElementById('expected-'+concept).value,
    same_as: document.getElementById('same-'+concept).value,
    comment: document.getElementById('comment-'+concept).value
  }));
  for (const row of answers) {
    if (row.same_as !== '' && (!/^\d+$/.test(row.same_as) || +row.same_as >= REPORT_ID.n_components || +row.same_as === row.concept)) {
      document.getElementById('copy-status').textContent = `Same as must be another concept from 0 to ${REPORT_ID.n_components - 1}.`;
      return;
    }
  }
  const text = JSON.stringify({...REPORT_ID, revealed, answers}, null, 2);
  const output = document.getElementById('answers');
  output.hidden = false;
  output.value = text;
  try {
    await navigator.clipboard.writeText(text);
    document.getElementById('copy-status').textContent = 'Answers copied.';
  } catch (_) {
    output.select();
    document.getElementById('copy-status').textContent = 'Copy the selected answers below.';
  }
}
"""


def select_tiles(
    codes: np.ndarray,
    keys: pd.DataFrame,
    n_tiles: int = 8,
    min_total_quantile: float = 0.5,
) -> pd.DataFrame:
    """IBD share ranking, with explicit eligible filtering and stable ties.

    Unlike upstream's zero-score fallback, never fill a shortage with an
    ineligible tile. Zero-total codes carry no concept and are always excluded.
    """
    total = codes.sum(1)
    eligible = (total >= np.quantile(total, min_total_quantile)) & (total > 0)
    shares = code_share(codes)
    selected = []
    for k in range(codes.shape[1]):
        indices = np.flatnonzero(eligible)
        indices = indices[np.argsort(-shares[indices, k], kind="stable")]
        frame = keys.iloc[indices].assign(share=shares[indices, k], concept=k)
        frame = frame.drop_duplicates("slide_id").head(n_tiles)
        selected.append(frame.assign(rank=np.arange(len(frame))))
    return pd.concat(selected, ignore_index=True)


def tile_geometry(slide) -> dict:
    import openslide

    try:
        reader = openslide.OpenSlide(str(slide.path))
    except openslide.OpenSlideUnsupportedFormatError:
        if slide.level != 0:
            raise ValueError(
                f"Cannot read the trained pyramid level from {slide.path}"
            ) from None
        reader = openslide.ImageSlide(str(slide.path))
    with reader:
        properties = reader.properties
        physical = properties.get(openslide.PROPERTY_NAME_MPP_X)
        physical_y = properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if physical and physical_y:
            from rationai.masks import closest_level

            scanner_level = closest_level(reader, float(slide.card["mpp"]))
        elif len(reader.level_dimensions) == 1:
            scanner_level = 0
        else:
            raise ValueError(
                "Slide MPP is required to recover the original closest-MPP tiling level."
            )
        width, height = reader.level_dimensions[scanner_level]
        width0, height0 = reader.dimensions
        downsample = float(reader.level_downsamples[scanner_level])
        mpp_x = float(physical) * downsample if physical else float(slide.card["mpp"])
        mpp_y = (
            float(physical_y) * downsample if physical_y else float(slide.card["mpp"])
        )
    if not all(np.isfinite(v) and v > 0 for v in (downsample, mpp_x, mpp_y)):
        raise ValueError("Invalid source slide geometry.")
    return {
        "width": width,
        "height": height,
        "width0": width0,
        "height0": height0,
        "downsample": downsample,
        "scanner_level": scanner_level,
        "coordinate_system": "tiling_level",
        "tile_extent": int(slide.card["tile_extent"]),
        "stride": int(slide.card.get("stride", slide.card["tile_extent"])),
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp_source": "slide" if physical and physical_y else "configured_nominal",
    }


def read_rgb_tile(slide, x: int, y: int):
    import openslide
    from PIL import Image

    geometry = tile_geometry(slide)

    try:
        reader = openslide.OpenSlide(str(slide.path))
    except openslide.OpenSlideUnsupportedFormatError:
        if slide.level != 0:
            raise
        reader = openslide.ImageSlide(str(slide.path))
    with reader:
        rgba = reader.read_region(
            (int(x * geometry["downsample"]), int(y * geometry["downsample"])),
            geometry["scanner_level"],
            (int(slide.card["tile_extent"]),) * 2,
        )
        return Image.alpha_composite(
            Image.new("RGBA", rgba.size, (255, 255, 255)), rgba
        ).convert("RGB")


def write_presence_mask(values, keys, geometry, low, high, path: Path) -> None:
    """Use MammaPrint's existing overlap assembler and TIFF writer.

    Coordinates and assembly both use the trained tiling level.
    HeatmapAssembler expects stride-grid positions scaled by its common divisor.
    Padding handles partial final cells; the resulting TIFF is cropped exactly
    to the source pyramid level and carries that level's physical resolution.
    """
    import pyvips
    import torch
    from rationai.masks import HeatmapAssembler, write_big_tiff

    tile, stride = geometry["tile_extent"], geometry["stride"]
    divisor = math.gcd(tile, stride)
    width = math.ceil(geometry["width"] / divisor) * divisor
    height = math.ceil(geometry["height"] / divisor) * divisor
    positions = keys[["x", "y"]].to_numpy()
    grid = np.rint(positions / stride)
    if np.any(np.abs(positions - grid * stride) > 1.01):
        raise ValueError("Tile coordinates do not align with the trained stride grid.")
    xs, ys = torch.from_numpy((grid * divisor).astype(np.int64)).unbind(1)
    assembler = HeatmapAssembler(width, height, tile, tile, stride, stride)
    coverage = HeatmapAssembler(width, height, tile, tile, stride, stride)
    assembler.update(torch.as_tensor(values, dtype=torch.float32), xs, ys)
    coverage.update(torch.ones(len(values)), xs, ys)
    averaged = assembler.compute().numpy()
    covered = coverage.compute().numpy() > 0
    scaled = np.clip((averaged - low) / max(high - low, 1e-12), 0, 1)
    pixels = np.where(covered, 1 + np.rint(254 * scaled), 0).astype(np.uint8)
    image = pyvips.Image.new_from_array(pixels).resize(divisor, kernel="nearest")
    image = image.crop(0, 0, geometry["width"], geometry["height"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(image, path, mpp_x=geometry["mpp_x"], mpp_y=geometry["mpp_y"])


def tile_viewport(x: float, y: float, geometry: dict) -> dict:
    """IBD viewport formula with all lengths expressed in the same tiling level."""
    tile = geometry["tile_extent"]
    return {
        "zoomLevel": geometry["width"] / (4 * tile),
        "point": {
            "x": (x + tile / 2) / geometry["width"],
            "y": (y + tile / 2) / geometry["width"],
        },
    }


def row_urls(
    row,
    geometries,
    paths,
    output_dir,
    source_run_id,
    dictionary_id,
    publication,
    n_concepts=16,
):
    """IBD xOpat multi-slide session pattern; only one concept layer is visible."""
    from report_fast import XopatSession, heatmap_layer
    from report_fast.mlflow import artifact_data_id

    n_concepts = validate_n_components(n_concepts)
    concept = int(row.iloc[0].concept)
    if concept < 0 or concept >= n_concepts or not (row.concept == concept).all():
        raise ValueError("xOpat row concept does not match n_components.")
    session = XopatSession()
    viewports = []
    for tile in row.itertuples():
        geometry = geometries[tile.slide_id]
        visualization = session.add_visualization(str(tile.slide_id))
        session.add_background(
            str(paths[tile.slide_id]),
            name=str(tile.slide_id),
            visualization_index=visualization,
            protocol="wsi_service",
        )
        for k in range(n_concepts):
            relative = f"masks/concept_{k:02d}/{tile.slide_id}.tiff"
            mask = (
                artifact_data_id(
                    relative, publication["run_id"], publication["experiment_id"]
                )
                if publication
                else str(output_dir / relative)
            )
            index = session.add_layer(
                {
                    **heatmap_layer(mask, name=f"Concept {k}", opacity=0.5),
                    "visible": k == concept,
                },
                visualization=visualization,
            )
            session.data[int(index)]["protocol"] = "wsi_service"
        viewports.append(tile_viewport(tile.x, tile.y, geometry))
    urls = []
    for index, viewport in enumerate(viewports):
        session.params = {"viewport": viewport, "activeBackgroundIndex": index}
        urls.append(session.url())
    return urls


def render_naming_html(manifest, selected, images, effects, presence, fidelity):
    n_components = validate_n_components(manifest.get("n_components", 16))
    for frame in (selected, effects, presence):
        if not frame.empty:
            if "concept" not in frame:
                raise ValueError("Report rows must declare their concept.")
            for value in frame.concept:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or not 0 <= value < n_components
                ):
                    raise ValueError("Report concept is outside declared n_components.")
    if len(images) != len(selected):
        raise ValueError("Report images do not match selected tiles.")

    def escape(value):
        return html.escape(str(value), quote=True)

    report_id = {
        key: manifest.get(key) for key in ("dictionary_id", "source_run_id", "split")
    }
    report_id["n_components"] = n_components
    rows = []
    for k in range(n_components):
        examples = selected[selected.concept == k]
        tiles = "".join(
            f'<a href="{escape(row.xopat_url)}" target="_blank" rel="noopener"><img width="160" src="data:image/jpeg;base64,{images[i]}" alt="Concept {k}"><small>{escape(row.slide_id)} {escape(getattr(row, "footprint", ""))}</small></a>'
            for i, row in examples.iterrows()
        )
        shortage = (
            f"<p>Only {len(examples)} eligible slides: fewer than 8.</p>"
            if len(examples) < 8
            else ""
        )
        labels = (
            examples[["slide_id", "labels"]].to_html(index=False, escape=True)
            if "labels" in examples
            else ""
        )
        numbers = (
            effects[effects.concept == k].to_html(index=False, escape=True)
            if not effects.empty
            else "No evaluated effects."
        )
        prevalence = (
            presence[presence.concept == k].to_html(index=False, escape=True)
            if not presence.empty
            else ""
        )
        options = '<option value="">Choose</option>' + "".join(
            f"<option>{option}</option>" for option in ("A", "B", "no effect", "unsure")
        )
        rows.append(f"""<section><h2>Concept {k}</h2><div class="tiles">{tiles}</div>{shortage}
<label>Name <input id="name-{k}"></label>
<label><input type="checkbox" id="unclear-{k}">Unclear or mixed</label>
<label>Expected luminal type <select class="expected" id="expected-{k}">{options}</select></label>
<label>Same as concept <input id="same-{k}" type="number" min="0" max="{n_components - 1}"></label>
<label>Comment <textarea id="comment-{k}"></textarea></label>
<div class="model-info">{labels}{numbers}{prevalence}</div></section>""")
    fixture = (
        '<p class="warning">FIXTURE ONLY — synthetic/local smoke test, not a validated cohort result.</p>'
        if manifest.get("fixture_only")
        else ""
    )
    fidelity_html = (
        fidelity.to_html(index=False, escape=True)
        if not fidelity.empty
        else "Fidelity not available."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MammaPrint concept naming: K={n_components}</title>
<style>body{{font:15px system-ui;margin:24px}}body.blind .model-info{{display:none}}section{{border-top:1px solid #bbb;padding:16px 0}}.tiles{{display:flex;gap:8px;flex-wrap:wrap}}.tiles a{{display:flex;flex-direction:column}}label{{display:inline-block;margin:8px}}textarea{{vertical-align:middle}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px}}.warning{{color:#a00}}#answers{{width:95%;height:240px}}</style></head>
<body class="blind"><h1>MammaPrint: {n_components} visual concepts</h1>{fixture}
<p>Patterns extracted from frozen Virchow2 embeddings. Name them from the images before revealing labels and model effects. Each row uses distinct slides. Click a tile to inspect the slide and concept-presence layer in xOpat (VPN required).</p>
<p>Expected luminal type is a proposed breast-specific question. A pathologist must confirm its wording. Model index effects are shown separately after reveal.</p>
<p>Reloading or closing this page loses answers. Copy them regularly. Revealing results freezes expected-type answers. This is procedural blinding, not access control.</p>
<button id="reveal" onclick="revealResults()">Show labels and effects</button>
<button onclick="copyAnswers()">Copy your answers</button><span id="copy-status"></span><textarea id="answers" hidden readonly></textarea>
<div class="model-info"><h2>Fidelity and interpretation</h2>{fidelity_html}
<p>Effects compare two-sided concept changes against reconstructed-model output, not the original embeddings. Classification effects are percentage points of P(A) in group A or P(B) in group B; regression uses raw index units. Presence is relative concept frequency. Direction disagreement is flagged. These are model responses, not biological causal effects.</p></div>
{"".join(rows)}<script>const REPORT_ID={json.dumps(report_id).replace("<", chr(92) + "u003c")};{FORM_SCRIPT}</script></body></html>"""


def build_report(
    source,
    dictionary_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    *,
    publication=None,
) -> Path:
    manifest = json.loads((evaluation_dir / "manifest.json").read_text())
    dictionary_manifest = json.loads((dictionary_dir / "manifest.json").read_text())
    if (
        manifest["dictionary_id"] != dictionary_manifest["dictionary_id"]
        or manifest["source_run_id"] != source.run_id
    ):
        raise ValueError("Report source run or dictionary identity mismatch.")
    n_components = dictionary_n_components(dictionary_manifest)
    load_dictionary(
        dictionary_dir
    )  # Validate dimensions and checksum before using its codes.
    if (
        validate_n_components(manifest.get("n_components", n_components))
        != n_components
    ):
        raise ValueError("Evaluation n_components does not match the dictionary.")
    manifest = {**manifest, "n_components": n_components}
    output_dir.mkdir(parents=True, exist_ok=True)
    split = manifest["split"]
    dataset = source_dataset(source, split)
    files = [
        evaluation_dir / "coefficients" / f"{name}.parquet"
        for name in sorted(dataset.slides["name"])
    ]
    counts = [len(pd.read_parquet(path, columns=["row_index"])) for path in files]
    codes = np.lib.format.open_memmap(
        output_dir / ".report-codes.npy",
        mode="w+",
        dtype=np.float32,
        shape=(sum(counts), n_components),
    )
    keys, offset = [], 0
    for path, count in zip(files, counts, strict=True):
        frame = pd.read_parquet(path)
        codes[offset : offset + count] = read_codes(frame, n_components)
        keys.append(frame[["slide_id", "row_index", "x", "y"]])
        offset += count
    keys = pd.concat(keys, ignore_index=True)
    selected = select_tiles(codes, keys)
    shares = code_share(codes)
    low, high = np.quantile(shares, [0.5, 0.999], axis=0)
    geometries, paths, images_by_key = {}, {}, {}
    chosen = set(selected.slide_id)
    for slide in iter_slides(source, split, dataset):
        if slide.slide_id not in chosen:
            continue
        geometry = tile_geometry(slide)
        geometries[slide.slide_id], paths[slide.slide_id] = geometry, slide.path
        frame = pd.read_parquet(
            evaluation_dir / "coefficients" / f"{slide.slide_id}.parquet"
        )
        pd.testing.assert_frame_equal(
            frame[["slide_id", "row_index", "x", "y"]], slide.keys, check_dtype=False
        )
        local_shares = code_share(read_codes(frame, n_components))
        for k in range(n_components):
            write_presence_mask(
                local_shares[:, k],
                slide.keys,
                geometry,
                low[k],
                high[k],
                output_dir / f"masks/concept_{k:02d}/{slide.slide_id}.tiff",
            )
        for row in selected[selected.slide_id == slide.slide_id].itertuples():
            buffer = io.BytesIO()
            read_rgb_tile(slide, row.x, row.y).save(buffer, format="JPEG", quality=90)
            images_by_key[row.Index] = base64.b64encode(buffer.getvalue()).decode()
    selected["xopat_url"] = ""
    for _, row in selected.groupby("concept"):
        selected.loc[row.index, "xopat_url"] = row_urls(
            row,
            geometries,
            paths,
            output_dir,
            source.run_id,
            manifest["dictionary_id"],
            publication,
            n_concepts=n_components,
        )
    labels = (
        pd.read_parquet(evaluation_dir / "slide_outputs.parquet")
        .groupby("slide_id")
        .apply(
            lambda rows: "; ".join(
                f"{row.target}: {row.label:g}" for row in rows.itertuples()
            ),
            include_groups=False,
        )
    )
    selected["labels"] = selected.slide_id.map(labels)
    selected["footprint"] = selected.slide_id.map(
        lambda sid: (
            f"~{geometries[sid]['tile_extent'] * geometries[sid]['mpp_x']:.0f} µm ({geometries[sid]['mpp_source']})"
        )
    )
    selected.to_parquet(output_dir / "representative_tiles.parquet", index=False)
    effects = pd.read_parquet(evaluation_dir / "concept_effects.parquet")
    presence = pd.read_parquet(evaluation_dir / "concept_presence.parquet")
    fidelity = pd.read_parquet(evaluation_dir / "fidelity.parquet")
    path = output_dir / f"naming_k{n_components}.html"
    path.write_text(
        render_naming_html(
            manifest,
            selected,
            [images_by_key[i] for i in selected.index],
            effects,
            presence,
            fidelity,
        )
    )
    save_json(
        output_dir / "report_manifest.json",
        {
            **manifest,
            "geometry": geometries,
            "mask_scale_low": low.tolist(),
            "mask_scale_high": high.tolist(),
            "template": "adapted_ibd_report_plus_slack_brief",
            "publication": publication,
            "notes": [
                "Exact original naming template pending Adam",
                "Live xOpat alignment not validated locally",
            ],
        },
    )
    del codes
    (output_dir / ".report-codes.npy").unlink()
    return path
