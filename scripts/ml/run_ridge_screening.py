"""Build the slide cache or run one stage of the CPU ridge experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits

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
    root = argparse.ArgumentParser(description=__doc__)
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
    return root


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
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
