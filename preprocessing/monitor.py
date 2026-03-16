"""Lightweight resource monitor that appends CPU / RAM / GPU stats to a CSV.

Usage:
    monitor = ResourceMonitor("/tmp/resource_usage.csv", interval=60)
    monitor.start()

    monitor.log_phase("read_slides", tile_count=0)
    # ... do work ...
    monitor.log_phase("tissue_filter", tile_count=120000)
    # ... do work ...

    monitor.stop()

Reads directly from /proc so it works inside containers without psutil.
GPU stats are collected via nvidia-smi and silently skipped when unavailable.
"""

import csv
import os
import subprocess
import threading
import time
from pathlib import Path

_COLUMNS = [
    "timestamp",
    "elapsed_s",
    "phase",
    "tile_count",
    "cpu_pct",
    "ram_used_gb",
    "ram_total_gb",
    "gpu_index",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
]


def _cpu_times() -> tuple[float, float]:
    """Return (total_jiffies, idle_jiffies) from /proc/stat."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    # user nice system idle iowait irq softirq steal
    values = [int(v) for v in parts[1:]]
    total = sum(values)
    idle = values[3] + values[4]  # idle + iowait
    return total, idle


def _memory_info() -> tuple[float, float]:
    """Return (used_gb, total_gb) from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            # Values are in kB
            info[key.strip()] = int(rest.strip().split()[0])
    total = info["MemTotal"] / (1024**2)
    available = info["MemAvailable"] / (1024**2)
    return round(total - available, 2), round(total, 2)


def _gpu_stats() -> list[dict[str, str]]:
    """Return per-GPU utilisation and memory via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        idx, util, mem_used, mem_total = [v.strip() for v in line.split(",")]
        gpus.append(
            {
                "gpu_index": idx,
                "gpu_util_pct": util,
                "gpu_mem_used_mb": mem_used,
                "gpu_mem_total_mb": mem_total,
            }
        )
    return gpus


class ResourceMonitor:
    """Background thread that periodically writes resource usage to a CSV.

    Supports named phases so you can see which pipeline stage each sample
    belongs to and how many tiles have been processed so far.
    """

    def __init__(self, path: str | Path, interval: float = 60.0) -> None:
        self._path = Path(path)
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._phase: str = ""
        self._tile_count: int | str = ""
        self._start_time: float = 0.0

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)

    def log_phase(self, phase: str, tile_count: int | None = None) -> None:
        """Update the current phase label and optional tile count.

        This is also immediately written as a row so you get an exact
        timestamp for each phase transition, independent of the sampling
        interval.
        """
        with self._lock:
            self._phase = phase
            self._tile_count = tile_count if tile_count is not None else ""

        elapsed = round(time.monotonic() - self._start_time)
        count_str = str(tile_count) if tile_count is not None else ""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[monitor] phase={phase}  tiles={count_str}  elapsed={elapsed}s")

        # Write an immediate sample so phase boundaries are visible in the CSV
        self._write_sample_now()

    # -- internals ------------------------------------------------------------

    def _write_sample_now(self) -> None:
        """Take a snapshot and append it to the CSV (called from main thread)."""
        try:
            cpu_pct = ""  # No delta available for instant sample
            ram_used, ram_total = _memory_info()
            gpus = _gpu_stats()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = round(time.monotonic() - self._start_time)

            with self._lock:
                phase = self._phase
                tile_count = self._tile_count

            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", newline="") as f:
                writer = csv.writer(f)
                if gpus:
                    for gpu in gpus:
                        writer.writerow(
                            [
                                ts, elapsed, phase, tile_count, cpu_pct,
                                ram_used, ram_total,
                                gpu["gpu_index"], gpu["gpu_util_pct"],
                                gpu["gpu_mem_used_mb"], gpu["gpu_mem_total_mb"],
                            ]
                        )
                else:
                    writer.writerow(
                        [ts, elapsed, phase, tile_count, cpu_pct, ram_used, ram_total,
                         "", "", "", ""]
                    )
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:  # noqa: BLE001
            print(f"[monitor] error writing phase sample: {exc}")

    def _run(self) -> None:
        prev_total, prev_idle = _cpu_times()
        write_header = not self._path.exists() or self._path.stat().st_size == 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_COLUMNS)
                f.flush()

            while not self._stop.wait(self._interval):
                try:
                    self._sample(writer, f, prev_total, prev_idle)
                except Exception as exc:  # noqa: BLE001
                    print(f"[monitor] error: {exc}")
                prev_total, prev_idle = _cpu_times()

    def _sample(
        self,
        writer: csv.writer,  # type: ignore[type-arg]
        f: object,
        prev_total: float,
        prev_idle: float,
    ) -> None:
        cur_total, cur_idle = _cpu_times()
        dt = cur_total - prev_total
        cpu_pct = round(100.0 * (1.0 - (cur_idle - prev_idle) / dt), 1) if dt else 0.0

        ram_used, ram_total = _memory_info()
        gpus = _gpu_stats()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = round(time.monotonic() - self._start_time)

        with self._lock:
            phase = self._phase
            tile_count = self._tile_count

        if gpus:
            for gpu in gpus:
                writer.writerow(  # type: ignore[union-attr]
                    [
                        ts, elapsed, phase, tile_count, cpu_pct,
                        ram_used, ram_total,
                        gpu["gpu_index"], gpu["gpu_util_pct"],
                        gpu["gpu_mem_used_mb"], gpu["gpu_mem_total_mb"],
                    ]
                )
        else:
            writer.writerow(  # type: ignore[union-attr]
                [ts, elapsed, phase, tile_count, cpu_pct, ram_used, ram_total,
                 "", "", "", ""]
            )

        f.flush()  # type: ignore[union-attr]
        os.fsync(f.fileno())  # type: ignore[union-attr]
