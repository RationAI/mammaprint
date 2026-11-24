"""Probe the allocated CUDA device and sample resources without a CUDA context."""

import argparse
import ctypes
import json
import os
import re
import shutil
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
    # Prefer the exact CUDA UUID. It can be unprefixed or reported as GPU- by CUDA
    # even when NVML calls the same UUID a MIG instance.
    match = re.fullmatch(r"(?:GPU-|MIG-)?([0-9a-f]{8}(?:-[0-9a-f]+){4})", uuid, re.I)
    if match:
        prefixes = (
            ("MIG-", "GPU-")
            if "MIG" in str(info["name"])
            or any(value.startswith("MIG-") for value in visible)
            else ("GPU-", "MIG-")
        )
        return [prefix + match[1] for prefix in prefixes]
    return [
        value
        for value in visible
        if value.startswith(("MIG-", "GPU-")) and "," not in value
    ]


class DeviceMemory:
    def __init__(self, info: dict[str, object]):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.uuid = ""
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        for uuid in device_uuids(info):
            if (
                self.library.nvmlDeviceGetHandleByUUID(
                    uuid.encode(), ctypes.byref(self.handle)
                )
                == 0
            ):
                self.uuid = uuid
                return
        self.close()
        raise RuntimeError("NVML could not address the allocated CUDA device by UUID")

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
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(probe_cuda()))
