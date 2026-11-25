"""Probe the allocated CUDA device and sample resources without a CUDA context."""

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


class MemoryInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


def probe_cuda() -> dict[str, object]:
    import torch

    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f"Expected exactly one visible CUDA device, got {count}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "count": count,
        "name": properties.name,
        "cuda_uuid": str(getattr(properties, "uuid", "")),
        "cuda_total_bytes": properties.total_memory,
        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", ""),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }


def device_uuids(info: dict[str, object]) -> list[str]:
    uuid = str(info["cuda_uuid"])
    visible = [
        str(info[key]) for key in ("cuda_visible_devices", "nvidia_visible_devices")
    ]
    # An explicit MIG assignment identifies the instance even when CUDA reports
    # the parent GPU's UUID on a MIG-enabled device.
    explicit = [
        value for value in visible if value.startswith("MIG-") and "," not in value
    ]
    match = re.fullmatch(r"(?:GPU-|MIG-)?([0-9a-f]{8}(?:-[0-9a-f]+){4})", uuid, re.I)
    if match:
        prefixes = (
            ("MIG-", "GPU-")
            if "MIG" in str(info["name"]) or explicit
            else ("GPU-", "MIG-")
        )
        return list(
            dict.fromkeys([*explicit, *(prefix + match[1] for prefix in prefixes)])
        )
    return [
        value
        for value in visible
        if value.startswith(("MIG-", "GPU-")) and "," not in value
    ]


def mig_child_uuid(listing: str, parent_uuid: str) -> str | None:
    children = []
    under_parent = False
    for line in listing.splitlines():
        if line.startswith("GPU "):
            under_parent = line.rstrip().endswith(f"(UUID: {parent_uuid})")
        elif under_parent and line.lstrip().startswith("MIG "):
            match = re.search(r"\(UUID: (MIG-[^()\s]+)\)\s*$", line)
            if match:
                children.append(match[1])
    return children[0] if len(children) == 1 else None


class DeviceMemory:
    def __init__(self, info: dict[str, object]):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.uuid = ""
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        candidates = device_uuids(info)
        mig = "MIG" in str(info["name"]) or any(
            str(info[key]).startswith("MIG-")
            for key in ("cuda_visible_devices", "nvidia_visible_devices")
        )
        try:
            for candidate in candidates:
                uuid = candidate
                if mig and candidate.startswith("GPU-"):
                    listing = subprocess.check_output(
                        ["nvidia-smi", "-L"],
                        text=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL,
                    )
                    uuid = mig_child_uuid(listing, candidate)
                    if uuid is None:
                        continue
                if (
                    self.library.nvmlDeviceGetHandleByUUID(
                        uuid.encode(), ctypes.byref(self.handle)
                    )
                    == 0
                ):
                    self.uuid = uuid
                    return
            raise RuntimeError(
                "NVML could not address the allocated CUDA device by UUID"
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.close()
            raise RuntimeError(str(error)) from error

    def read(self) -> MemoryInfo:
        info = MemoryInfo()
        result = self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
        if result != 0:
            raise RuntimeError(f"NVML memory sample failed with status {result}")
        return info

    def close(self) -> None:
        self.library.nvmlShutdown()


def read_int(*paths: str) -> int | None:
    for path in paths:
        try:
            return int(Path(path).read_text())
        except (OSError, ValueError):
            continue
    return None


def cpu_usage_usec() -> int | None:
    try:
        counters = dict(
            line.split()
            for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
        )
        return int(counters["usage_usec"])
    except (OSError, ValueError, KeyError):
        value = read_int(
            "/sys/fs/cgroup/cpuacct/cpuacct.usage",
            "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
        )
        return value // 1000 if value is not None else None


class ResourceSampler:
    def __init__(self, output: Path, scratch: Path, info: dict[str, object]):
        self.output = output
        self.scratch = scratch
        self.next_sample = 0.0
        self.device = None
        self.device_error = None
        self.samples = 0
        self.cuda_samples = 0
        self.peaks: dict[str, int] = {}
        self.minimum_scratch_free: int | None = None
        self.previous_cpu: tuple[float, int] | None = None
        self.peak_cpu_cores = 0.0
        try:
            self.device = DeviceMemory(info)
        except (OSError, RuntimeError) as error:
            self.device_error = str(error)

    def sample(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self.next_sample:
            return
        self.next_sample = now + 2.0
        metrics = {
            "at": datetime.now(UTC).isoformat(),
            "host_memory_bytes": read_int(
                "/sys/fs/cgroup/memory.current",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            ),
            "cpu_usage_usec": cpu_usage_usec(),
            "shm_used_bytes": shutil.disk_usage("/dev/shm").used
            if os.path.isdir("/dev/shm")
            else None,
            "scratch_filesystem_free_bytes": shutil.disk_usage(self.scratch).free,
            "cuda_uuid": self.device.uuid if self.device else None,
            "cuda_used_bytes": None,
            "cuda_total_bytes": None,
            "cuda_error": self.device_error,
        }
        if self.device:
            try:
                memory = self.device.read()
                metrics.update(
                    cuda_used_bytes=memory.used, cuda_total_bytes=memory.total
                )
            except RuntimeError as error:
                metrics["cuda_error"] = str(error)
                self.device_error = str(error)
        with (self.output / "resources.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")
        self.samples += 1
        self.cuda_samples += int(metrics["cuda_used_bytes"] is not None)
        for key in (
            "cuda_used_bytes",
            "cuda_total_bytes",
            "host_memory_bytes",
            "shm_used_bytes",
        ):
            value = metrics[key]
            if isinstance(value, int):
                self.peaks[key] = max(value, self.peaks.get(key, 0))
        free = metrics["scratch_filesystem_free_bytes"]
        if isinstance(free, int):
            self.minimum_scratch_free = (
                min(self.minimum_scratch_free, free)
                if self.minimum_scratch_free is not None
                else free
            )
        cpu = metrics["cpu_usage_usec"]
        if isinstance(cpu, int):
            if self.previous_cpu is not None:
                previous_time, previous_cpu = self.previous_cpu
                elapsed = now - previous_time
                if elapsed >= 0.5:
                    self.peak_cpu_cores = max(
                        self.peak_cpu_cores,
                        (cpu - previous_cpu) / (elapsed * 1_000_000),
                    )
            self.previous_cpu = (now, cpu)

    def summary(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "cuda_samples": self.cuda_samples,
            "cuda_uuid": self.device.uuid if self.device else None,
            "cuda_error": self.device_error,
            "peaks": self.peaks,
            "peak_cpu_cores": round(self.peak_cpu_cores, 2)
            if self.previous_cpu
            else None,
            "minimum_scratch_filesystem_free_bytes": self.minimum_scratch_free,
        }

    def close(self) -> None:
        if self.device:
            self.device.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--probe", action="store_true")
    operation.add_argument(
        "--sample-memory", type=Path, help="CUDA probe JSON from a running batch"
    )
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe_cuda()))
    else:
        device = DeviceMemory(
            json.loads(args.sample_memory.read_text(encoding="utf-8"))
        )
        try:
            while True:
                memory = device.read()
                sample = {
                    "at": datetime.now(UTC).isoformat(),
                    "uuid": device.uuid,
                    "used_bytes": memory.used,
                    "total_bytes": memory.total,
                }
                print(json.dumps(sample), flush=True)
                time.sleep(2.0)
        finally:
            device.close()
