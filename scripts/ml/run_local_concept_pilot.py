"""Bounded read-only remote downloads and LOCAL frozen concept smoke evaluation.

The default is a side-effect-free plan. All result paths must be outside Git.
Only --execute reads remote metadata, downloads <=2 complete validation WSIs,
fits a tiny training-subset dictionary and evaluates complete validation bags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen


REPO = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024


class ByteCapExceededError(ValueError):
    """An archive or total transfer/extraction budget was exhausted."""


@dataclass
class ByteBudget:
    limit: int
    used: int = 0

    def add(self, size):
        self.used += size
        if self.used > self.limit:
            raise ByteCapExceededError(
                "Combined byte cap exceeded; partial data removed."
            )


def verify_file(path: Path, expected: str, size: int | None = None):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Cached asset must be a regular file.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    if digest.hexdigest() != expected or (
        size is not None and path.stat().st_size != size
    ):
        raise ValueError("Cached asset checksum/size mismatch; refusing reuse.")
    return path


def download_bounded(url: str, destination: Path, cap: int, budget: ByteBudget):
    """Count bytes even on aborted transfers; never retain a partial file."""
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() or partial.exists():
        raise ValueError("Download destination already exists.")
    received = 0
    try:
        with urlopen(url, timeout=90) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (
                int(length) < 0
                or int(length) > cap
                or int(length) > budget.limit - budget.used
            ):
                raise ByteCapExceededError(
                    "Advertised Content-Length exceeds byte cap."
                )
            with partial.open("xb") as stream:
                while True:
                    remaining = min(cap - received, budget.limit - budget.used)
                    if remaining <= 0:
                        if length is not None and received == int(length):
                            break
                        raise ByteCapExceededError(
                            "Streaming byte cap reached before confirmed EOF."
                        )
                    block = response.read(min(MIB, remaining))
                    if not block:
                        break
                    received += len(block)
                    budget.add(len(block))
                    if received > cap:
                        raise ByteCapExceededError(
                            "Per-archive byte cap exceeded; partial data removed."
                        )
                    stream.write(block)
            if length is not None and received != int(length):
                raise ValueError("Downloaded byte count differs from Content-Length.")
        partial.rename(destination)
        return received
    finally:
        partial.unlink(missing_ok=True)


def _zip_is_dir(entry):
    return (
        entry.filename.replace("\\", "/").endswith("/")
        or stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFDIR
    )


def extract_safe(archive: Path, destination: Path, cap: int, budget: ByteBudget):
    """Extract to a new directory, rejecting unsafe names/types and zip bombs."""
    if destination.exists():
        raise ValueError("Extraction destination must be new.")
    written = 0
    try:
        with zipfile.ZipFile(archive) as source:
            members, names = [], set()
            for entry in source.infolist():
                name = entry.filename.replace("\\", "/")
                path = PurePosixPath(name)
                mode = entry.external_attr >> 16
                if (
                    not name
                    or path.is_absolute()
                    or ".." in path.parts
                    or ":" in name
                    or "\x00" in name
                    or mode & 0o170000 not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or entry.flag_bits & 1
                ):
                    raise ValueError("Unsafe or unsupported ZIP entry.")
                normalized = str(path).casefold()
                if normalized in names or normalized == ".":
                    raise ValueError("ZIP path collision.")
                names.add(normalized)
                members.append((entry, path))
            files = {
                str(path).casefold()
                for entry, path in members
                if not _zip_is_dir(entry)
            }
            if any(
                any(
                    str(parent).casefold() in files
                    for parent in path.parents
                    if str(parent) != "."
                )
                for _, path in members
            ):
                raise ValueError("ZIP file/directory collision.")
            total = sum(entry.file_size for entry, _ in members)
            if total > cap or total > budget.limit - budget.used:
                raise ByteCapExceededError("ZIP expanded size exceeds byte cap.")
            destination.mkdir(mode=0o700, parents=True)
            for entry, path in members:
                target = destination.joinpath(*path.parts)
                if _zip_is_dir(entry):
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with source.open(entry) as incoming, target.open("xb") as outgoing:
                    while block := incoming.read(min(MIB, cap - written + 1)):
                        written += len(block)
                        budget.add(len(block))
                        if written > cap:
                            raise ByteCapExceededError("Expanded byte cap exceeded.")
                        outgoing.write(block)
        return written
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def check_identity(run, experiment, run_id, experiment_id, experiment_name):
    info = run["info"]
    tags = {tag["key"]: tag["value"] for tag in run["data"].get("tags", [])}
    revision = tags.get("mlflow.source.git.commit", "")
    if (
        info.get("run_id") != run_id
        or str(info.get("experiment_id")) != experiment_id
        or info.get("status") != "FINISHED"
        or str(experiment.get("experiment_id")) != experiment_id
        or experiment.get("name") != experiment_name
        or re.fullmatch(r"[a-fA-F0-9]{40}", revision) is None
    ):
        raise ValueError(
            "Source run/experiment identity, status or exact training revision mismatch."
        )
    return revision


def selected_split_audit(assets, memberships):
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if memberships[left] & memberships[right]:
            raise ValueError("Original artifact slide memberships overlap.")
    for asset in assets:
        if asset["slide_id"] not in memberships[asset["split"]]:
            raise ValueError(
                "Selected bag does not belong to its claimed original split."
            )
    return {
        "scope": "original artifact memberships; selected complete train/val bags only",
        "slide_disjoint": True,
        "patient_key": None,
        "patient_audit": "unverified: original patient/case mapping unavailable",
        "test_bags_downloaded_or_scored": 0,
        "original_split_counts": {
            split: len(ids) for split, ids in memberships.items()
        },
    }


def json_get(base, route, **params):
    with urlopen(
        base.rstrip("/") + route + "?" + urlencode(params), timeout=60
    ) as response:
        # Metadata only, bounded independently from WSI transfer budgets.
        data = response.read(16 * MIB + 1)
    if len(data) > 16 * MIB:
        raise ValueError("Remote metadata response exceeds 16 MiB.")
    return json.loads(data)


def parse_artifact(uri):
    match = re.fullmatch(
        r"mlflow-artifacts:/([0-9]+)/([A-Za-z0-9_-]+)/artifacts/(.+)", uri
    )
    if not match or any(part in (".", "..", "") for part in match[3].split("/")):
        raise ValueError("Expected an explicit MLflow artifact URI.")
    return match.groups()


def membership(base, uri):
    _, run_id, directory = parse_artifact(uri)
    names, token = set(), None
    while True:
        params = {"run_id": run_id, "path": directory}
        if token:
            params["page_token"] = token
        response = json_get(base, "/api/2.0/mlflow/artifacts/list", **params)
        for item in response.get("files", []):
            path = PurePosixPath(item["path"])
            if (
                not item.get("is_dir")
                and path.suffix == ".parquet"
                and str(path.parent) == directory
            ):
                names.add(path.stem)
        token = response.get("next_page_token")
        if not token:
            break
    if not names:
        raise ValueError("Original split artifact membership listing is empty.")
    return names


def load_assets(path):
    assets = json.loads(path.read_text())
    if not isinstance(assets, list) or len(assets) != 4:
        raise ValueError(
            "Pilot requires exactly two train and two complete validation bags."
        )
    seen = set()
    for asset in assets:
        sid = asset["slide_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in seen:
            raise ValueError("Duplicate/unsafe slide identifier.")
        seen.add(sid)
        if (
            re.fullmatch(r"[a-f0-9]{64}", asset["sha256"]) is None
            or not 0 < asset["bytes"] <= 32 * MIB
        ):
            raise ValueError(
                "Small embedding asset checksum/size required (<=32 MiB each)."
            )
        parse_artifact(asset["source_uri"])
    if sorted(a["split"] for a in assets) != ["train", "train", "val", "val"]:
        raise ValueError(
            "Pilot requires two train and two validation bags; no test bags."
        )
    return assets


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "tracking-uri",
        "wsi-base-url",
        "source-run-id",
        "expected-experiment-id",
        "expected-experiment-name",
        "assets-manifest",
        "output-root",
        "cache-root",
        "source-config",
        "source-config-sha256",
        "checkpoint",
        "checkpoint-sha256",
        "slides",
        "slides-sha256",
        "slides-uri",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--skip-wsi",
        action="store_true",
        help="Numerical smoke only; explicitly skip local visual/mask validation.",
    )
    p.add_argument("--archive-byte-cap", type=int, default=512 * MIB)
    p.add_argument("--total-byte-cap", type=int, default=1024 * MIB)
    p.add_argument("--expanded-byte-cap", type=int, default=2048 * MIB)
    p.add_argument("--total-expanded-byte-cap", type=int, default=4096 * MIB)
    return p


def validate_args(args):
    for name in ("tracking_uri", "wsi_base_url"):
        parsed = urlparse(getattr(args, name))
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Service bases must be explicit HTTPS URLs without credentials/query/fragment."
            )
    if not args.wsi_base_url.rstrip("/").endswith("/v3"):
        raise ValueError("WSI service base URL must end in /v3.")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.source_run_id)
        or not args.expected_experiment_id.isdigit()
    ):
        raise ValueError(
            "Explicit safe source run and numeric experiment identity required."
        )
    for name in ("source_config", "checkpoint", "slides"):
        if re.fullmatch(r"[a-f0-9]{64}", getattr(args, name + "_sha256")) is None:
            raise ValueError("SHA256 pins are required for cached source assets.")
    parse_artifact(args.slides_uri)
    for name in (
        "archive_byte_cap",
        "total_byte_cap",
        "expanded_byte_cap",
        "total_expanded_byte_cap",
    ):
        if getattr(args, name) <= 0:
            raise ValueError("Byte caps must be positive.")
    for name in ("output_root", "cache_root"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if path.is_relative_to(REPO):
            raise ValueError(
                "Private data/result directories must be outside this repository."
            )
        setattr(args, name, path)
    if (
        args.output_root == args.cache_root
        or args.output_root.is_relative_to(args.cache_root)
        or args.cache_root.is_relative_to(args.output_root)
    ):
        raise ValueError(
            "Output and cache roots must be separate non-nested directories."
        )


def prepare_local_sources(source, cache_root, chosen, assets):
    """Adapt complete local bags and reject any production-loader exclusions."""
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    from ml.data.datasets.single_scale import SingleScaleDataset

    config = source.config
    cached = cache_root / "embeddings"
    cached.mkdir(mode=0o700)
    inspected = {}
    for split in ("train", "val"):
        directory = cached / split
        directory.mkdir(mode=0o700)
        for asset in (a for a in assets if a["split"] == split):
            target = directory / (asset["slide_id"] + ".parquet")
            shutil.copyfile(asset["local_path"], target)
            verify_file(target, asset["sha256"], asset["bytes"])
        mapping = cache_root / (split + "-mapping.csv")
        rows = chosen[
            chosen.name.isin(a["slide_id"] for a in assets if a["split"] == split)
        ]
        rows.to_csv(mapping, index=False)
        # Only the local dataset is instantiated; original config and URI identity remain intact.
        local_card = OmegaConf.to_container(config.dataset.levels, resolve=True)
        for value in local_card.values():
            value["uris"] = {split: str(directory)}
        source.datasets[split] = SingleScaleDataset(
            split, local_card, mapping, str(config.label_mode)
        )
        expected = {a["slide_id"]: a for a in assets if a["split"] == split}
        dataset = source.datasets[split]
        loaded = dataset.slides["name"].astype(str)
        if len(expected) != 2 or len(loaded) != 2 or set(loaded) != set(expected):
            raise ValueError(
                f"{split} loaded roster differs from selected original bags: "
                f"expected exactly two complete labelled bags, retained {len(loaded)}."
            )
        if {p.stem for p in dataset.embeddings_dir.glob("*.parquet")} != set(expected):
            raise ValueError(
                f"{split} cached parquet roster differs from selected bags."
            )
        inspected[split] = {}
        for sid, asset in expected.items():
            count = pq.read_metadata(dataset.embeddings_dir / f"{sid}.parquet").num_rows
            supplied = asset.get("rows")
            if count <= 0:
                raise ValueError(
                    f"{split} selected bags require nonzero complete row counts."
                )
            if supplied is not None and (
                type(supplied) is not int or supplied <= 0 or count != supplied
            ):
                raise ValueError(
                    f"{split} complete bag row count differs from inspected asset manifest."
                )
            inspected[split][sid] = count
    return inspected


def execute(args, assets):
    # Heavy model/native imports occur only after explicit execution opt-in.
    sys.path.insert(0, str(REPO))
    import pandas as pd
    from omegaconf import OmegaConf

    from ml.concept_explainability.evaluation import (
        evaluate_concepts,
        fit_dictionary,
        save_json,
    )
    from ml.concept_explainability.model import SourceRun, verify_training_revision
    from ml.concept_explainability.report import build_report, tile_geometry

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("Output root must be empty; existing results are immutable.")
    for name in ("source_config", "checkpoint", "slides"):
        verify_file(Path(getattr(args, name)), getattr(args, name + "_sha256"))
    config = OmegaConf.load(args.source_config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    run = json_get(
        args.tracking_uri, "/api/2.0/mlflow/runs/get", run_id=args.source_run_id
    )["run"]
    experiment = json_get(
        args.tracking_uri,
        "/api/2.0/mlflow/experiments/get",
        experiment_id=args.expected_experiment_id,
    )["experiment"]
    revision = check_identity(
        run,
        experiment,
        args.source_run_id,
        args.expected_experiment_id,
        args.expected_experiment_name,
    )
    verify_training_revision(revision, config=config)
    if len(config.dataset.levels) != 1:
        raise ValueError("Pilot supports a single original embedding level.")
    _, card = next(iter(config.dataset.levels.items()))
    memberships = {
        split: membership(args.tracking_uri, str(card.uris[split]))
        for split in ("train", "val", "test")
    }
    for asset in assets:
        expected = (
            str(card.uris[asset["split"]]).rstrip("/")
            + "/"
            + asset["slide_id"]
            + ".parquet"
        )
        if asset["source_uri"] != expected:
            raise ValueError(
                "Selected artifact does not match original training config URI."
            )
        verify_file(Path(asset["local_path"]), asset["sha256"], asset["bytes"])
    audit = selected_split_audit(assets, memberships)
    metadata = pd.read_parquet(args.slides)
    if "path" not in metadata:
        raise ValueError("Original slides metadata needs persisted source paths.")
    metadata["name"] = metadata.path.map(lambda path: Path(path).stem)
    chosen = metadata[metadata.name.isin(a["slide_id"] for a in assets)].copy()
    if len(chosen) != 4 or chosen.name.duplicated().any():
        raise ValueError("Selected original metadata rows must be unique and complete.")
    # Do not promote the original table's id to a patient/case identifier.
    chosen = chosen.drop(
        columns=[
            c for c in ("id", "patient_id", "case_id", "record_num") if c in chosen
        ]
    )
    os.umask(0o077)
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output_root.chmod(0o700)
    args.cache_root.chmod(0o700)
    provenance = {
        "smoke_only": True,
        "fixture_only": True,
        "scientific_validation": False,
        "selection": "explicit supplied roster, selected previously by smallest parquet bytes within split; runner verifies membership, not original selection ranking",
        "assets_manifest_sha256": hashlib.sha256(
            Path(args.assets_manifest).read_bytes()
        ).hexdigest(),
        "source_run_id": args.source_run_id,
        "training_sha": revision,
        "tracking_uri": args.tracking_uri,
        "expected_experiment_id": args.expected_experiment_id,
        "expected_experiment_name": args.expected_experiment_name,
        "wsi_base_url": args.wsi_base_url,
        "source_config_sha256": args.source_config_sha256,
        "checkpoint_sha256": args.checkpoint_sha256,
        "labels_geometry_uri": args.slides_uri,
        "labels_geometry_sha256": args.slides_sha256,
        "labels_provenance_limit": "persisted original tiling table; byte-identical training label CSV unverified",
        "split_audit": audit,
        "selected_assets": assets,
        "original_memberships": {
            split: sorted(ids) for split, ids in memberships.items()
        },
        "dictionary_scope": "smoke-only training subset; never the full corpus dictionary",
        "validation_scope": "two complete original validation bags; descriptive SD only",
        "execution": {
            "device": "cpu",
            "log_to_mlflow": False,
            "n_components": 16,
            "use_ray": False,
        },
        "caps": {
            name: getattr(args, name)
            for name in (
                "archive_byte_cap",
                "total_byte_cap",
                "expanded_byte_cap",
                "total_expanded_byte_cap",
            )
        },
        "wsi": [],
        "report_status": "pending",
        "status": "running",
    }
    save_json(args.output_root / "pilot_manifest.json", provenance)
    source = SourceRun(
        args.source_run_id,
        revision,
        config,
        Path(args.checkpoint),
        args.checkpoint_sha256,
    )
    provenance["loaded_bag_rows"] = prepare_local_sources(
        source, args.cache_root, chosen, assets
    )
    save_json(args.output_root / "pilot_manifest.json", provenance)
    compressed, expanded = (
        ByteBudget(args.total_byte_cap),
        ByteBudget(args.total_expanded_byte_cap),
    )
    if not args.skip_wsi:
        for row in source.datasets["val"].slides.itertuples():
            entry = {"slide_id": row.name, "status": "pending"}
            archive = args.cache_root / (row.name + ".zip")
            extraction = args.cache_root / (row.name + "-wsi")
            # Persisted metadata paths are source identity; map mount prefix to service root.
            original = "/" + str(row.path).replace("\\", "/").lstrip("/")
            if not original.startswith("/mnt/data/") or not original.endswith(".mrxs"):
                raise ValueError("Expected original /mnt/data/... MRXS source path.")
            service_id = original.removeprefix("/mnt/")
            entry["service_slide_id"] = service_id
            try:
                url = (
                    args.wsi_base_url.rstrip("/")
                    + "/slides/download?"
                    + urlencode({"slide_id": service_id})
                )
                entry["archive_bytes"] = download_bounded(
                    url, archive, args.archive_byte_cap, compressed
                )
                entry["expanded_bytes"] = extract_safe(
                    archive, extraction, args.expanded_byte_cap, expanded
                )
                paths = list(extraction.rglob("*.mrxs"))
                if len(paths) != 1 or paths[0].stem != row.name:
                    raise ValueError("ZIP does not contain the expected unique MRXS.")
                sidecars = paths[0].with_suffix("")
                if (
                    not (sidecars / "Index.dat").is_file()
                    or not (sidecars / "Slidedat.ini").is_file()
                ):
                    raise ValueError("MRXS sidecar directory is incomplete.")
                import openslide

                with openslide.OpenSlide(str(paths[0])) as reader:
                    reader.read_region((0, 0), 0, (1, 1))
                source.datasets["val"].slides.loc[
                    source.datasets["val"].slides.name == row.name, "path"
                ] = str(paths[0])
                entry["local_path"] = str(paths[0])
                entry["status"] = "complete_opened"
            except Exception as error:
                # Numerical validation remains useful when full WSI exceeds the fixed cap.
                entry["status"] = "unavailable"
                entry["failure_type"] = type(error).__name__
                shutil.rmtree(extraction, ignore_errors=True)
            finally:
                archive.unlink(missing_ok=True)
            provenance["wsi"].append(entry)
            save_json(args.output_root / "pilot_manifest.json", provenance)
    from ml.concept_explainability.data import iter_slides

    # Verify exact persisted tiling geometry before reporting local WSI-backed masks.
    if len([w for w in provenance["wsi"] if w["status"] == "complete_opened"]) == 2:
        for slide in iter_slides(source, "val"):
            geometry = tile_geometry(slide)
            original = chosen.set_index("name").loc[slide.slide_id]
            if (
                geometry["scanner_level"] != int(original.level)
                or geometry["width"] != int(original.extent_x)
                or geometry["height"] != int(original.extent_y)
                or abs(geometry["downsample"] - float(original.downsample)) > 1e-6
            ):
                raise ValueError(
                    "Local WSI geometry does not match original tiling metadata."
                )
    fit = OmegaConf.load(REPO / "configs/concepts.yaml").fit
    dictionary, evaluation = (
        args.output_root / "dictionary",
        args.output_root / "validation",
    )
    fit_dictionary(source, fit, dictionary, use_ray=False)
    evaluate_concepts(source, dictionary, "val", evaluation, device="cpu")
    for directory in (dictionary, evaluation):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pilot"] = provenance
        save_json(path, manifest)
    if len([w for w in provenance["wsi"] if w["status"] == "complete_opened"]) == 2:
        build_report(source, dictionary, evaluation, evaluation, publication=None)
        html = evaluation / "naming_k16.html"
        html.write_text(
            html.read_text().replace(
                '<body class="blind">',
                '<body class="blind"><p class="warning">REAL-ASSET SMOKE ONLY — two training bags, two validation bags; patient separation unverified. Local xOpat paths are not remotely accessible.</p>',
            )
        )
        provenance["report_status"] = (
            "local_WSI_report_built; remote xOpat alignment unverified"
        )
    else:
        provenance["report_status"] = (
            "skipped: two complete local WSIs unavailable or explicitly skipped"
        )
    provenance["transfer_bytes"] = compressed.used
    provenance["expanded_bytes"] = expanded.used
    provenance["status"] = "completed_smoke_only"
    save_json(args.output_root / "pilot_manifest.json", provenance)
    report_manifest = evaluation / "report_manifest.json"
    if report_manifest.exists():
        report = json.loads(report_manifest.read_text())
        report["pilot"] = provenance
        save_json(report_manifest, report)
    for directory in (dictionary, evaluation):
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["pilot"] = provenance
        save_json(directory / "manifest.json", manifest)
    return {
        "mode": "executed",
        "smoke_only": True,
        "report_status": provenance["report_status"],
    }


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    assets = load_assets(Path(args.assets_manifest))
    if args.execute:
        result = execute(args, assets)
    else:
        result = {
            "mode": "plan",
            "smoke_only": True,
            "selected_bags": 4,
            "max_validation_wsis": 0 if args.skip_wsi else 2,
            "selected_embedding_bytes": sum(a["bytes"] for a in assets),
            "archive_byte_cap": args.archive_byte_cap,
            "total_byte_cap": args.total_byte_cap,
            "actions": [
                "validate pinned cache and live run identity",
                "audit original split membership metadata",
                "download bounded complete validation WSI ZIPs",
                "fit K16 training-subset fixture dictionary",
                "evaluate complete validation bags locally",
                "report only with two complete local WSIs",
            ],
            "writes_downloads_or_computation": False,
        }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
