"""Run named commands with bounded concurrency in one GPU allocation."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from scripts.ml.training_batch_resources import ResourceSampler


POLL_SECONDS = 0.05
SHUTDOWN_SECONDS = 3


class RunSpec(TypedDict):
    id: str
    argv: list[str]


class BatchSpec(TypedDict):
    version: int
    batch_id: str
    max_parallel: int
    runs: list[RunSpec]


class RunStatus(TypedDict):
    id: str
    state: str
    log_path: str
    started_at: str | None
    ended_at: str | None
    exit_code: int | None
    error: str | None


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}", value
    ):
        raise ValueError(f"Invalid batch, attempt, execution or run ID: {value!r}")
    return value


def validate_spec(value: object) -> BatchSpec:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("Expected a version 1 batch spec")
    batch_id = identifier(value.get("batch_id"))
    max_parallel = value.get("max_parallel")
    if type(max_parallel) is not int or max_parallel < 1:
        raise ValueError("max_parallel must be a positive integer")
    entries = value.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("runs must be a nonempty list")

    runs: list[RunSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each run must be an object with an ID and argv")
        run_id = identifier(entry.get("id"))
        if run_id.casefold() in seen:
            raise ValueError(f"Duplicate run ID: {run_id}")
        seen.add(run_id.casefold())
        argv = entry.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or not argv[0]
        ):
            raise ValueError(f"Run {run_id} requires a nonempty argv list of strings")
        runs.append({"id": run_id, "argv": argv})
    return {
        "version": 1,
        "batch_id": batch_id,
        "max_parallel": max_parallel,
        "runs": runs,
    }


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def signal_group(process: subprocess.Popen, signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def stop_processes(processes: list[subprocess.Popen]) -> None:
    groups = [process for process in processes if signal_group(process, signal.SIGTERM)]
    deadline = time.monotonic() + SHUTDOWN_SECONDS
    while groups and time.monotonic() < deadline:
        for process in groups:
            process.poll()
        groups = [process for process in groups if signal_group(process, 0)]
        if groups:
            time.sleep(POLL_SECONDS)
    for process in groups:
        signal_group(process, signal.SIGKILL)
    for process in processes:
        process.wait()


def run_batch(
    spec: BatchSpec,
    output: Path,
    scratch: Path,
    attempt_id: str,
    execution_id: str,
    gpu_info: dict[str, object] | None = None,
) -> int:
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    write_json(output / "spec.json", spec)
    started = timestamp()
    statuses: dict[str, RunStatus] = {
        run["id"]: {
            "id": run["id"],
            "state": "queued",
            "log_path": f"{run['id']}/console.log",
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "error": None,
        }
        for run in spec["runs"]
    }

    def save_status(*, finished: bool = False) -> None:
        snapshot = {
            "batch_id": spec["batch_id"],
            "attempt_id": attempt_id,
            "execution_id": execution_id,
            "started_at": started,
            "ended_at": timestamp() if finished else None,
            "runs": list(statuses.values()),
        }
        write_json(output / "status.json", snapshot)
        if finished:
            print(f"training_batch_result={json.dumps(snapshot)}", flush=True)

    def finish(
        run_id: str, state: str, exit_code: int | None, error: str | None = None
    ) -> None:
        statuses[run_id].update(
            state=state, ended_at=timestamp(), exit_code=exit_code, error=error
        )
        print(f"{run_id}: {state} (exit={exit_code})", flush=True)
        save_status()
        run_scratch = scratch / run_id
        if run_scratch.exists():
            try:
                shutil.rmtree(run_scratch)
            except OSError as error:
                print(f"{run_id}: could not clean scratch: {error}", flush=True)

    child_env = dict(
        os.environ,
        TRAIN_BATCH_ID=spec["batch_id"],
        TRAIN_ATTEMPT_ID=attempt_id,
        TRAIN_EXECUTION_ID=execution_id,
        PYTHONUNBUFFERED="1",
    )
    stopping_signal = 0

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal stopping_signal
        stopping_signal = stopping_signal or signum

    old_handlers = {
        sig: signal.signal(sig, request_shutdown)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    pending = deque(spec["runs"])
    active: dict[str, subprocess.Popen] = {}
    sampler = None
    try:
        save_status()
        if gpu_info is not None:
            write_json(output / "gpu.json", gpu_info)
            sampler = ResourceSampler(output, scratch, gpu_info)
        while (pending or active) and not stopping_signal:
            if sampler:
                sampler.sample()
            for run_id, process in list(active.items()):
                result = process.poll()
                if result is not None:
                    # Descendants left by an exited trainer cannot occupy the next slot.
                    signal_group(process, signal.SIGKILL)
                    del active[run_id]
                    finish(run_id, "succeeded" if result == 0 else "failed", result)

            while (
                pending and len(active) < spec["max_parallel"] and not stopping_signal
            ):
                run = pending.popleft()
                run_id = run["id"]
                run_dir = output / run_id
                run_scratch = scratch / run_id
                env = dict(
                    child_env,
                    TRAIN_RUN_ID=run_id,
                    TRAIN_RUN_DIR=str(run_dir),
                    TMPDIR=str(run_scratch),
                )
                try:
                    run_dir.mkdir()
                    run_scratch.mkdir()
                    with (run_dir / "console.log").open("wb") as log:
                        if stopping_signal:
                            pending.appendleft(run)
                            break
                        active[run_id] = subprocess.Popen(
                            run["argv"],
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env,
                            start_new_session=True,
                        )
                except OSError as error:
                    finish(run_id, "failed", None, str(error))
                    continue
                statuses[run_id].update(state="running", started_at=timestamp())
                print(f"{run_id}: running ({run_dir / 'console.log'})", flush=True)
                save_status()
            if active and not stopping_signal:
                time.sleep(POLL_SECONDS)
    finally:
        try:
            stop_processes(list(active.values()))
            for run_id, process in active.items():
                finish(run_id, "interrupted", process.returncode)
            for run in pending:
                finish(run["id"], "not_started", None)
            save_status(finished=True)
            if sampler:
                sampler.sample(force=True)
                summary = sampler.summary()
                write_json(output / "resources-summary.json", summary)
                print(f"training_batch_resources={json.dumps(summary)}", flush=True)
        finally:
            if sampler:
                sampler.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    if stopping_signal:
        return 128 + stopping_signal
    return int(any(status["state"] != "succeeded" for status in statuses.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument(
        "--gpu-info", type=Path, help="Probe output for optional resource logging"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = validate_spec(json.loads(args.spec.read_text(encoding="utf-8")))
        attempt = identifier(args.attempt_id)
        execution = identifier(args.execution_id)
        relative = Path(spec["batch_id"]) / attempt / execution
        output = args.output_root.resolve() / relative
        scratch = args.scratch_root.resolve() / relative
        if output == scratch:
            raise ValueError("Output and scratch directories must be distinct")
        if args.dry_run:
            print(
                json.dumps(
                    {"spec": spec, "output": str(output), "scratch": str(scratch)},
                    indent=2,
                )
            )
            return 0
        info = (
            json.loads(args.gpu_info.read_text(encoding="utf-8"))
            if args.gpu_info
            else None
        )
        return run_batch(spec, output, scratch, attempt, execution, info)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
