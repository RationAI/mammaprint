r"""Fit a CPU ridge model or run the MammaPrint ridge screening workflow.

Standalone fit with a repository data card:
  uv run --frozen python -m scripts.ml.run_ridge_screening fit \
    --data-mapping data_mapping.csv --level 3 --mask tissue_only --output ridge-l3

Custom dataset with an explicit directory or MLflow artifact URI:
  uv run --frozen python -m scripts.ml.run_ridge_screening fit \
    --data-mapping cohort.csv --level 4 --mask custom_mask --source embeddings/ \
    --target-column response --embedding-dim 768 --feature ln-mean-std --output ridge-custom

Use --cache-dir CACHE to fit from an existing slide-statistics cache. The CSV
needs path, record_num, split (train, val, or test), and a continuous target
column (mammaprint_index by default). Embedding files are <slide-stem>.parquet
with an embedding vector column. The level and mask select one precomputed
embedding source per invocation.

Fit uses labeled train and validation slides with available features. Test
slides are reserved. By default it selects alpha by grouped CV MAE and refits
on the full development cohort. Use --alpha to set a fixed penalty.

The ln features normalize each tile across its embedding coordinates using
the saved LayerNorm epsilon. The std features use population standard
deviation across tiles. The model.json file contains the coefficients,
intercept, and ordered feature blocks. Predict with their dot product plus
the intercept.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits

from ml.ridge_fit import FEATURE_COLUMNS, FitOptions, fit_standalone, resolve_source
from preprocessing.slide_statistics import build_cache, source_inventory


ROOT = Path(__file__).resolve().parents[2]
SCREENS = ("levels", "multilevel", "statistics", "masks", "learning-curve", "all")
PATH_FIELDS = (
    "data_mapping",
    "cards_dir",
    "scratch_dir",
    "reuse_cache",
    "cache_dir",
    "sanity_dir",
    "screen_dir",
    "lock_dir",
    "historical_reference",
    "test_registry",
)


def config_from_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a configuration mapping")
    config.setdefault("cards_dir", ROOT / "configs/data/embedded")
    for key in PATH_FIELDS:
        if config.get(key) is not None:
            config[key] = (path.parent / config[key]).resolve()
    for name, spec in config.get("sources", {}).items():
        uri = spec.get("uri") if isinstance(spec, dict) else spec
        if uri and not urlsplit(uri).scheme:
            uri = str((path.parent / uri).resolve())
            config["sources"][name] = (
                {**spec, "uri": uri} if isinstance(spec, dict) else uri
            )
    return config


def parser() -> argparse.ArgumentParser:
    description, _, examples = (__doc__ or "").partition("\n\n")
    root = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    modes = root.add_subparsers(dest="mode", required=True)
    for mode in ("cache", "sanity", "screen", "lock", "test"):
        sub = modes.add_parser(mode)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--dry-run", action="store_true")
        if mode == "screen":
            sub.add_argument("--screen", choices=SCREENS, default="all")
        if mode == "cache":
            sub.add_argument("--pilot-slides", type=int)
    fit = modes.add_parser(
        "fit",
        help="Fit on a single dataset, level, and mask",
        description="Fit a standalone closed-form ridge model on development slides.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fit.add_argument("--data-mapping", type=Path, required=True)
    fit.add_argument("--level", type=int, required=True)
    fit.add_argument("--mask", required=True)
    fit.add_argument("--output", type=Path, required=True)
    inputs = fit.add_mutually_exclusive_group()
    inputs.add_argument(
        "--source", help="Directory or MLflow artifact with slide embedding Parquets"
    )
    inputs.add_argument("--cache-dir", type=Path, help="Reuse a full statistics cache")
    inputs.add_argument(
        "--cards-dir",
        type=Path,
        default=ROOT / "configs/data/embedded",
        help="Root of the mask/l<level>.yaml data cards",
    )
    fit.add_argument("--target-column", default="mammaprint_index")
    fit.add_argument("--feature", choices=FEATURE_COLUMNS, default="raw-mean")
    fit.add_argument(
        "--embedding-dim",
        type=int,
        default=2560,
        help="Vector width when building a cache",
    )
    fit.add_argument(
        "--scratch-dir", type=Path, help="Scratch directory for embedding downloads"
    )
    fit.add_argument("--alpha", type=float, help="Fixed penalty in place of grouped CV")
    fit.add_argument(
        "--folds", type=int, default=3, help="Group folds used when tuning alpha"
    )
    fit.add_argument("--seed", type=int, default=0, help="Fold assignment seed")
    fit.add_argument("--blas-threads", type=int, default=1)
    fit.add_argument("--dry-run", action="store_true")
    return root


def run_fit_command(args: argparse.Namespace) -> None:
    options = FitOptions(
        mapping=args.data_mapping.resolve(),
        level=args.level,
        mask=args.mask,
        feature=args.feature,
        target_column=args.target_column,
        embedding_dim=args.embedding_dim,
        alpha=args.alpha,
        folds=args.folds,
        seed=args.seed,
        source=args.source,
        cards_dir=args.cards_dir.resolve(),
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        scratch_dir=args.scratch_dir.resolve() if args.scratch_dir else None,
    )
    options.validate()
    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be positive")
    source = None if options.cache_dir else resolve_source(options)
    output = args.output.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": "fit",
                    "data_mapping": str(options.mapping),
                    "source_key": options.key,
                    "source": source,
                    "cache_dir": str(options.cache_dir or output / "cache"),
                    "feature": options.feature,
                    "target_column": options.target_column,
                    "alpha": options.alpha,
                    "output": str(output / "model.json"),
                },
                indent=2,
            )
        )
        return
    with threadpool_limits(limits=args.blas_threads):
        print(fit_standalone(options, output, source))


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        if args.mode == "fit":
            run_fit_command(args)
            return
        config = config_from_file(args.config)
        output = args.output.resolve()
        if args.mode == "cache" and args.pilot_slides is not None:
            if args.pilot_slides < 1:
                raise ValueError("--pilot-slides must be positive")
            config["pilot_slides"] = args.pilot_slides
        if args.dry_run:
            preview = {
                "command": args.mode,
                "output": str(output),
                "data_mapping": str(config["data_mapping"]),
                "pilot_slides": config.get("pilot_slides"),
            }
            if args.mode == "cache":
                preview["sources"] = source_inventory(config)
            if args.mode == "screen":
                from ml.ridge_experiment import screen_arms

                preview["cache_dir"] = str(config["cache_dir"])
                preview["screens"] = screen_arms(config, args.screen)
            print(json.dumps(preview, indent=2, default=str))
            return

        from ml.ridge_experiment import (
            evaluate_test,
            lock_model,
            run_sanity,
            run_screen,
        )

        threads = int(config.get("blas_threads", 1))
        if threads < 1:
            raise ValueError("blas_threads must be positive")
        with threadpool_limits(limits=threads):
            if args.mode == "cache":
                print(build_cache(config, output))
                return
            output.mkdir(parents=True, exist_ok=True)
            with (output / ".write.lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.mode == "sanity":
                    result = run_sanity(config, output)
                    print(output / "sanity.json")
                    if not result["passed"]:
                        raise ValueError(
                            "Historical sanity check is unresolved. See sanity.json."
                        )
                elif args.mode == "screen":
                    result = run_screen(config, output, args.screen)
                    print(output / "report.md")
                    if (args.screen == "all" and not result["winner"]) or (
                        args.screen != "all" and args.screen in result["errors"]
                    ):
                        raise ValueError(
                            "Requested evaluation is infeasible. See report.md."
                        )
                elif args.mode == "lock":
                    lock_model(config, output)
                    print(output / "lock.json")
                else:
                    evaluate_test(config, output)
                    print(output / "test_metrics.json")
    except (OSError, KeyError, ValueError) as exc:
        cli.exit(1, f"Ridge screening: {exc}\n")


if __name__ == "__main__":
    main()
