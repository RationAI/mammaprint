"""Offline K16/8/32 smoke comparison from an audited cached pilot handoff.

Default: JSON-only plan, no writes/model imports. --execute uses existing local
inputs and fresh private outputs; network access is blocked for the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.ml.run_local_concept_pilot import (  # noqa: E402
    load_assets,
    selected_split_audit,
    verify_file,
)


COMPONENTS = (16, 8, 32)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def offline():
    """Fail closed on Python socket connections, datagrams and DNS resolution."""

    def denied(*args, **kwargs):
        raise RuntimeError(
            "This comparison is strictly offline; network access denied."
        )

    targets = [
        (socket, "getaddrinfo"),
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket.socket, "sendto"),
    ]
    if hasattr(socket.socket, "sendmsg"):
        targets.append((socket.socket, "sendmsg"))
    originals = [(obj, key, getattr(obj, key)) for obj, key in targets]
    try:
        for obj, key, _ in originals:
            setattr(obj, key, denied)
        yield
    finally:
        for obj, key, value in originals:
            setattr(obj, key, value)


def file_inventory(root):
    """Defined inventory: sorted relative paths, sizes, hashes; canonical JSON."""
    root = Path(root)
    files = []
    if root.is_symlink() or not root.is_dir():
        raise ValueError("WSI extraction must be an existing regular directory.")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Cached WSI symlinks are unsupported.")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "serialization": "UTF-8 JSON sorted by relative POSIX path; sort_keys=True,separators=(',',':')",
    }


def load_inputs(args):
    for name in ("reuse_inputs_sha256", "baseline_manifest_sha256"):
        if re.fullmatch(r"[a-f0-9]{64}", getattr(args, name)) is None:
            raise ValueError("Independent SHA256 pins are required.")
    reuse_path = verify_file(Path(args.reuse_inputs), args.reuse_inputs_sha256)
    reuse = json.loads(reuse_path.read_text())
    baseline_path = verify_file(
        Path(reuse["baseline_manifest"]), args.baseline_manifest_sha256
    )
    baseline = json.loads(baseline_path.read_text())
    verify_file(Path(reuse["assets_manifest"]), reuse["assets_manifest_sha256"])
    assets = load_assets(Path(reuse["assets_manifest"]))
    output = Path(args.output_root).expanduser().resolve()
    inputs = [reuse_path, baseline_path]
    inputs += [
        Path(reuse[k])
        for k in (
            "checkpoint",
            "source_config",
            "original_tiling_metadata",
            "assets_manifest",
            "baseline_dictionary",
            "baseline_validation",
        )
    ]
    inputs += [
        Path(p)
        for key in ("local_bags", "local_label_mapping")
        for p in reuse[key].values()
    ]
    inputs += [Path(w["extraction_root"]) for w in reuse["local_validation_wsis"]]
    if output.is_relative_to(REPO) or any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
        for p in inputs
    ):
        raise ValueError(
            "Private output must be outside Git and separate from every input."
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output root must be empty; existing results are immutable.")
    args.output_root = output
    return reuse, baseline, assets


def prepare_source(reuse, baseline, assets):
    """Validate complete cached inputs and adapt local datasets without transport."""
    import pandas as pd
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    from ml.concept_explainability.data import iter_slides
    from ml.concept_explainability.model import SourceRun, verify_training_revision
    from ml.data.datasets._sources import load_labeled_slides
    from ml.data.datasets.labels import LabelMode
    from ml.data.datasets.single_scale import SingleScaleDataset

    for key in (
        "source_run_id",
        "training_sha",
        "checkpoint_sha256",
        "source_config_sha256",
        "assets_manifest_sha256",
    ):
        if reuse[key] != baseline[key]:
            raise ValueError("Handoff identity conflicts with audited pilot manifest.")
    if baseline["status"] != "completed_smoke_only" or baseline["execution"] != {
        "device": "cpu",
        "log_to_mlflow": False,
        "n_components": 16,
        "use_ray": False,
    }:
        raise ValueError("Requires the completed frozen CPU K16 smoke pilot.")
    for key, sha_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("source_config", "source_config_sha256"),
        ("original_tiling_metadata", "original_tiling_metadata_sha256"),
        ("assets_manifest", "assets_manifest_sha256"),
    ):
        verify_file(Path(reuse[key]), reuse[sha_key])
    if reuse["original_tiling_metadata_sha256"] != baseline["labels_geometry_sha256"]:
        raise ValueError("Original tiling metadata pin changed.")
    if assets != baseline["selected_assets"]:
        raise ValueError("Selected asset manifest changed from the audited pilot.")
    memberships = {s: set(ids) for s, ids in baseline["original_memberships"].items()}
    split_audit = selected_split_audit(assets, memberships)
    config = OmegaConf.load(reuse["source_config"])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    verify_training_revision(reuse["training_sha"], config=config)
    if len(config.dataset.levels) != 1:
        raise ValueError(
            "Only the original single-level embedding source is supported."
        )
    source = SourceRun(
        reuse["source_run_id"],
        reuse["training_sha"],
        config,
        Path(reuse["checkpoint"]),
        reuse["checkpoint_sha256"],
    )
    metadata = pd.read_parquet(reuse["original_tiling_metadata"])
    metadata["name"] = metadata.path.map(lambda p: Path(p).stem)
    selected = metadata[metadata.name.isin(a["slide_id"] for a in assets)].copy()
    if len(selected) != 4 or selected.name.duplicated().any():
        raise ValueError(
            "Original selected metadata roster must contain four unique rows."
        )
    selected = selected.drop(
        columns=[
            c for c in ("id", "patient_id", "case_id", "record_num") if c in selected
        ]
    )
    level, card = next(iter(config.dataset.levels.items()))
    evidence = {
        "bag_rows": {},
        "mapping_sha256": {},
        "split_audit": split_audit,
        "wsi_inventory": [],
    }
    for split in ("train", "val"):
        expected = {a["slide_id"]: a for a in assets if a["split"] == split}
        directory = Path(reuse["local_bags"][split])
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or {p.stem for p in directory.glob("*.parquet")} != set(expected)
        ):
            raise ValueError(
                "Cached parquet roster differs from the audited selected split."
            )
        for sid, asset in expected.items():
            if (
                asset["source_uri"]
                != str(card.uris[split]).rstrip("/") + f"/{sid}.parquet"
            ):
                raise ValueError(
                    "Selected artifact URI differs from original source config."
                )
            path = verify_file(
                directory / f"{sid}.parquet", asset["sha256"], asset["bytes"]
            )
            count = pq.read_metadata(path).num_rows
            if (
                count <= 0
                or count != asset["rows"]
                or count != baseline["loaded_bag_rows"][split][sid]
            ):
                raise ValueError("Complete per-slide bag row count changed.")
        mapping = Path(reuse["local_label_mapping"][split])
        if mapping.is_symlink() or not mapping.is_file():
            raise ValueError("Local label mapping must be a regular file.")
        actual = pd.read_csv(mapping).sort_values("name").reset_index(drop=True)
        expected_mapping = (
            selected[selected.name.isin(expected)]
            .sort_values("name")
            .reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(
                actual.sort_index(axis=1),
                expected_mapping.sort_index(axis=1),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                "Cached label/geometry mapping differs from pinned original metadata."
            ) from exc
        # The normal constructor always invokes MLflow transport, even for a local
        # directory. Populate its local fields using the SAME production label loader.
        dataset = SingleScaleDataset.__new__(SingleScaleDataset)
        dataset.level, dataset.label_mode = (
            int(level),
            LabelMode(str(config.label_mode)),
        )
        dataset.embeddings_dir = directory
        dataset.slides = load_labeled_slides(mapping, dataset.label_mode)
        dataset.empty_slides = ()
        if (
            len(dataset.slides) != 2
            or dataset.slides.name.duplicated().any()
            or set(dataset.slides.name) != set(expected)
        ):
            raise ValueError(
                "Production label loading excluded/changed the selected roster."
            )
        source.datasets[split] = dataset
        evidence["bag_rows"][split] = {
            s.slide_id: len(s.embeddings) for s in iter_slides(source, split)
        }
        evidence["mapping_sha256"][split] = digest(mapping)
    if {s: sum(rows.values()) for s, rows in evidence["bag_rows"].items()} != {
        "train": 294,
        "val": 339,
    }:
        raise ValueError(
            "Comparison requires the same complete 294 train / 339 val tiles."
        )
    wsis = reuse["local_validation_wsis"]
    if len(wsis) != 2 or {w["slide_id"] for w in wsis} != set(
        evidence["bag_rows"]["val"]
    ):
        raise ValueError(
            "Exactly the two original cached validation WSIs are required."
        )
    for wsi in wsis:
        root, path = Path(wsi["extraction_root"]), Path(wsi["local_path"])
        inventory = file_inventory(root)
        if (
            len(inventory["files"]) != wsi["n_files"]
            or list(root.rglob("*.mrxs")) != [path]
            or path.stem != wsi["slide_id"]
            or not all(
                (path.with_suffix("") / n).is_file()
                for n in ("Index.dat", "Slidedat.ini")
            )
        ):
            raise ValueError("Existing complete MRXS/sidecar structure changed.")
        evidence["wsi_inventory"].append(
            {
                **wsi,
                "inventory": inventory,
                "old_tree_hash_status": "unverified: serialization unavailable",
            }
        )
        source.datasets["val"].slides.loc[
            source.datasets["val"].slides.name == wsi["slide_id"], "path"
        ] = str(path)
    return source, evidence


def check_original_outputs(expected, actual):
    import numpy as np

    keys = ["slide_id", "target"]
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise ValueError("Duplicate original output keys.")
    left, right = (f.set_index(keys).sort_index() for f in (expected, actual))
    if not left.index.equals(right.index) or not np.array_equal(
        left.original_raw.to_numpy(), right.original_raw.to_numpy()
    ):
        raise ValueError(
            "Original frozen-model outputs must match exactly across K and baseline."
        )


def native_outputs(source):
    import numpy as np
    import pandas as pd
    import torch

    from ml.concept_explainability.data import iter_slides
    from ml.concept_explainability.evaluation import model_bag
    from ml.concept_explainability.model import load_frozen_model, model_outputs

    module = load_frozen_model(source)
    before = {k: v.clone() for k, v in module.state_dict().items()}
    mode = str(source.config.label_mode)
    targets = ["luminal_a_probability"] if mode == "type" else ["mammaprint_index"]
    if mode == "both":
        targets = ["luminal_a_probability", "mammaprint_index"]
    rows = []
    with torch.inference_mode():
        for slide in iter_slides(source, "val"):
            bag = model_bag(module, slide, slide.embeddings, "cpu")
            native = module(bag)[0].reshape(-1).numpy()
            adapter = model_outputs(module, bag, mode).numpy()
            if not np.array_equal(native, adapter):
                raise ValueError("Native forward and concept adapter outputs differ.")
            rows += [
                {
                    "slide_id": slide.slide_id,
                    "target": target,
                    "original_raw": float(native[i]),
                }
                for i, target in enumerate(targets)
            ]
    if (
        module.training
        or any(p.requires_grad for p in module.parameters())
        or any(not torch.equal(before[k], v) for k, v in module.state_dict().items())
    ):
        raise ValueError("Source model must remain frozen/eval with unchanged state.")
    return pd.DataFrame(rows)


def execute(args, reuse, baseline, assets):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from ml.concept_explainability.data import iter_slides
    from ml.concept_explainability.evaluation import (
        evaluate_concepts,
        fit_dictionary,
        load_dictionary,
        read_codes,
        save_json,
    )
    from ml.concept_explainability.report import build_report, tile_geometry

    source, evidence = prepare_source(reuse, baseline, assets)
    for slide in iter_slides(source, "val"):
        geometry = tile_geometry(slide)
        row = pd.read_parquet(reuse["original_tiling_metadata"])
        row = row[row.path.map(lambda p: Path(p).stem) == slide.slide_id].iloc[0]
        if (
            geometry["scanner_level"] != int(row.level)
            or geometry["width"] != int(row.extent_x)
            or geometry["height"] != int(row.extent_y)
            or abs(geometry["downsample"] - float(row.downsample)) > 1e-6
        ):
            raise ValueError(
                "Cached WSI geometry differs from original tiling metadata."
            )
    baseline_dictionary = Path(reuse["baseline_dictionary"])
    baseline_validation = Path(reuse["baseline_validation"])
    dictionary_manifest = json.loads(
        (baseline_dictionary / "manifest.json").read_text()
    )
    validation_manifest = json.loads(
        (baseline_validation / "manifest.json").read_text()
    )
    load_dictionary(baseline_dictionary)
    for manifest in (dictionary_manifest, validation_manifest):
        if (
            any(manifest[k] != reuse[k] for k in ("source_run_id", "training_sha"))
            or manifest["dictionary_id"] != reuse["baseline_dictionary_id"]
            or not manifest["fixture_only"]
        ):
            raise ValueError("Baseline dictionary/evaluation identity changed.")
    if validation_manifest["checkpoint_sha256"] != reuse["checkpoint_sha256"]:
        raise ValueError("Baseline evaluation checkpoint pin changed.")
    fit = OmegaConf.create(dictionary_manifest["config"])
    if (
        OmegaConf.to_container(fit)
        != OmegaConf.to_container(OmegaConf.load(REPO / "configs/concepts.yaml").fit)
        or fit.n_components != 16
        or fit.nmf.random_state != 42
        or validation_manifest["step_sds"] != 0.25
    ):
        raise ValueError(
            "Baseline fit/intervention settings differ from unchanged defaults."
        )
    original = native_outputs(source)
    check_original_outputs(
        original, pd.read_parquet(baseline_validation / "slide_outputs.parquet")
    )
    os.umask(0o077)
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output_root.chmod(0o700)
    provenance = {
        "status": "running",
        "offline": True,
        "network_access": "blocked",
        "downloads": 0,
        "remote_writes": 0,
        "scientific_validation": False,
        "smoke_only": True,
        "fixture_only": True,
        "components": list(COMPONENTS),
        "source_run_id": source.run_id,
        "training_sha": source.training_sha,
        "checkpoint_sha256": source.checkpoint_sha256,
        "source_config_sha256": reuse["source_config_sha256"],
        "tiling_metadata_sha256": reuse["original_tiling_metadata_sha256"],
        "assets_manifest_sha256": reuse["assets_manifest_sha256"],
        "reuse_inputs_sha256": args.reuse_inputs_sha256,
        "baseline_manifest_sha256": args.baseline_manifest_sha256,
        "implementation_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "fit": OmegaConf.to_container(fit),
        "step_sds": 0.25,
        "device": "cpu",
        "original_output_parity": "exact native/adapter/baseline",
        "lineage": "Reused independently audited cached paths; live run identity not rechecked offline",
        "patient_audit": baseline["split_audit"]["patient_audit"],
        "evidence": evidence,
        "results": [],
    }
    manifest_path = args.output_root / "comparison_manifest.json"
    save_json(manifest_path, provenance)
    metrics = []
    try:
        for k in COMPONENTS:
            start = time.monotonic()
            folder = args.output_root / f"k{k}"
            dictionary, validation = folder / "dictionary", folder / "validation"
            candidate = OmegaConf.create(OmegaConf.to_container(fit))
            candidate.n_components = k
            fit_dictionary(source, candidate, dictionary, use_ray=False)
            fitted = json.loads((dictionary / "manifest.json").read_text())
            if k == 16 and fitted["dictionary_id"] != reuse["baseline_dictionary_id"]:
                raise ValueError(
                    "K16 refit dictionary differs from the audited baseline."
                )
            evaluate_concepts(
                source, dictionary, "val", validation, device="cpu", step_sds=0.25
            )
            check_original_outputs(
                original, pd.read_parquet(validation / "slide_outputs.parquet")
            )
            for split in ("train", "val"):
                for slide in iter_slides(source, split):
                    path = (
                        dictionary / "coefficients" / "train"
                        if split == "train"
                        else validation / "coefficients"
                    )
                    codes = pd.read_parquet(path / f"{slide.slide_id}.parquet")
                    read_codes(codes, k)
                    pd.testing.assert_frame_equal(
                        codes[["slide_id", "row_index", "x", "y"]],
                        slide.keys,
                        check_dtype=False,
                    )
            html = build_report(
                source, dictionary, validation, validation, publication=None
            )
            html.write_text(
                html.read_text().replace(
                    '<body class="blind">',
                    '<body class="blind"><p class="warning">REAL-ASSET SMOKE ONLY — two training bags, two validation bags; patient separation unverified. Local xOpat paths are not remotely accessible.</p>',
                )
            )
            evaluation = json.loads((validation / "manifest.json").read_text())
            elapsed = time.monotonic() - start
            for fidelity in evaluation["fidelity"]:
                values = [
                    evaluation["embedding_rmse"],
                    fidelity["raw_rmse"],
                    fidelity["display_rmse"],
                ]
                if (
                    not np.isfinite(values).all()
                    or evaluation["n_tiles"] != 339
                    or evaluation["n_slides"] != 2
                ):
                    raise ValueError("Invalid complete-bag comparison metrics.")
                metrics.append(
                    {
                        "k": k,
                        "target": fidelity["target"],
                        "dictionary_id": fitted["dictionary_id"],
                        "embedding_rmse": values[0],
                        "raw_rmse": values[1],
                        "display_rmse": values[2],
                        "decision_agreement": fidelity["label_agreement"],
                        "runtime_seconds": elapsed,
                    }
                )
            provenance["results"].append(
                {
                    "k": k,
                    "dictionary_id": fitted["dictionary_id"],
                    "naming_page": str(html),
                    "original_output_parity": "exact",
                    "runtime_seconds": elapsed,
                }
            )
            save_json(manifest_path, provenance)
        verify_file(Path(reuse["checkpoint"]), reuse["checkpoint_sha256"])
        pd.DataFrame(metrics).sort_values(["k", "target"]).to_csv(
            args.output_root / "comparison.csv", index=False
        )
        lines = [
            "# Local K comparison — smoke only",
            "",
            "|K|Target|Embedding RMSE|Raw RMSE|Displayed RMSE|Decision agreement|",
            "|---|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"|{r['k']}|{r['target']}|{r['embedding_rmse']:.8g}|{r['raw_rmse']:.8g}|{r['display_rmse']:.8g}|{r['decision_agreement']}|"
            for r in sorted(metrics, key=lambda r: (r["k"], r["target"]))
        ]
        lines += [
            "",
            "Same complete 294 train / 339 validation tiles; seed42 and all settings unchanged except K. K16 dictionary reproduced exactly; original native/adapter/baseline outputs agree exactly across all K.",
            "",
            "Embedding RMSE = sqrt(sum squared feature errors /339 tiles), without dividing by2560. Classification displayed RMSE is probability units; multiply by100 for percentage points. Decision agreement is original-v-reconstructed fidelity, not label accuracy. One seed/two validation slides cannot select optimal K, establish stability or scientific validity. Patient separation and historical training-label bytes remain unverified. Effects use per-K cohort SD calibration; effect magnitudes cannot rank K. Galleries necessarily fall short of eight slides. Local xOpat access/alignment remains unverified.",
            "",
        ]
        lines += [
            f"- [K{r['k']} naming page](k{r['k']}/validation/naming_k{r['k']}.html)"
            for r in provenance["results"]
        ]
        (args.output_root / "comparison.md").write_text("\n".join(lines))
        provenance["status"] = "completed_smoke_only"
    except Exception as error:
        provenance["status"] = "failed"
        provenance["failure_type"] = type(error).__name__
        raise
    finally:
        save_json(manifest_path, provenance)
    return {
        "mode": "executed",
        "status": provenance["status"],
        "output_root": str(args.output_root),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reuse-inputs",
        "reuse-inputs-sha256",
        "baseline-manifest-sha256",
        "output-root",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    with offline():
        reuse, baseline, assets = load_inputs(args)
        result = (
            execute(args, reuse, baseline, assets)
            if args.execute
            else {
                "mode": "plan",
                "components": list(COMPONENTS),
                "smoke_only": True,
                "offline": True,
                "asset_hashes_verified": False,
                "writes_or_computation": False,
                "actions": [
                    "verify pinned cached files, complete rosters/rows and geometry",
                    "reproduce K16 baseline then fit/evaluate/report K8 and K32 locally",
                    "require exact original frozen outputs across all K",
                ],
                "output_root": str(args.output_root),
            }
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
