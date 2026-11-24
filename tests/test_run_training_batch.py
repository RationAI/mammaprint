import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ml import training_batch_resources as resources
from scripts.ml.run_training_batch import run_batch, validate_spec


def command(tmp_path, runs, *, max_parallel=2, batch_id="test"):
    spec = {
        "version": 1,
        "batch_id": batch_id,
        "max_parallel": max_parallel,
        "runs": runs,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    return [
        sys.executable,
        "-m",
        "scripts.ml.run_training_batch",
        "--spec",
        str(spec_path),
        "--output-root",
        str(tmp_path / "output"),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--attempt-id",
        "a1",
        "--execution-id",
        "pod1",
    ]


def run(run_id, code, *args):
    return {"id": run_id, "argv": [sys.executable, "-c", code, *args]}


def output(tmp_path):
    return tmp_path / "output/test/a1/pod1"


def wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, path
        time.sleep(0.02)


def test_fills_vacant_slot_and_records_failures(tmp_path):
    gate = tmp_path / "gate"
    first = tmp_path / "first"
    latest = tmp_path / "latest"
    runs = [
        run(
            "long",
            """
import sys, time
from pathlib import Path
Path(sys.argv[1]).touch()
while not Path(sys.argv[2]).exists():
    time.sleep(.02)
""",
            str(first),
            str(gate),
        ),
        run(
            "fails",
            """
import sys, time
from pathlib import Path
while not Path(sys.argv[1]).exists():
    time.sleep(.02)
raise SystemExit(7)
""",
            str(first),
        ),
        {"id": "missing", "argv": [str(tmp_path / "missing-program")]},
        run(
            "next",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(latest),
        ),
    ]
    runner = subprocess.Popen(command(tmp_path, runs))
    try:
        wait_for(latest)
        records = json.loads((output(tmp_path) / "status.json").read_text())["runs"]
        assert [r["state"] for r in records[:3]] == ["running", "failed", "failed"]
        assert records[1]["exit_code"] == 7
        assert records[2]["exit_code"] is None
        assert records[2]["started_at"] is None
    finally:
        gate.touch()
        assert runner.wait(timeout=10) == 1
    records = json.loads((output(tmp_path) / "status.json").read_text())["runs"]
    assert [r["state"] for r in records] == [
        "succeeded",
        "failed",
        "failed",
        "succeeded",
    ]


def test_isolates_outputs_and_passes_literal_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-an-example")
    code = """
import json, os, sys
from pathlib import Path
Path(os.environ['TRAIN_RUN_DIR'], 'result.json').write_text(json.dumps([
    sys.argv[1], os.environ['TMPDIR'], os.environ['CUDA_VISIBLE_DEVICES'],
    os.environ['TRAIN_RUN_ID'], os.environ['TRAIN_EXECUTION_ID'],
]))
print('console output')
"""
    literal = "spaces; $(echo never executed)"
    args = command(tmp_path, [run("one", code, literal), run("two", code, literal)])
    result = subprocess.run([*args, "--dry-run"], check=True, capture_output=True)
    assert b"never executed" in result.stdout
    assert not output(tmp_path).exists()
    subprocess.run(args, check=True)
    for run_id in ("one", "two"):
        directory = output(tmp_path) / run_id
        actual = json.loads((directory / "result.json").read_text())
        assert actual == [
            literal,
            str(tmp_path / "scratch/test/a1/pod1" / run_id),
            "MIG-an-example",
            run_id,
            "pod1",
        ]
        assert (directory / "console.log").read_text() == "console output\n"
    assert subprocess.run(args, capture_output=True).returncode != 0


@pytest.mark.parametrize(
    "batch_id,max_parallel,ids",
    [
        ("../escape", 2, ["one"]),
        ("test", True, ["one"]),
        ("test", 0, ["one"]),
        ("test", 2, ["same", "same"]),
        ("test", 2, []),
        ("test", 2, ["../escape"]),
    ],
)
def test_rejects_invalid_specs_before_spawning(tmp_path, batch_id, max_parallel, ids):
    args = command(
        tmp_path,
        [run(i, "raise SystemExit(99)") for i in ids],
        batch_id=batch_id,
        max_parallel=max_parallel,
    )
    assert subprocess.run(args, capture_output=True).returncode == 2
    assert not (tmp_path / "output").exists()


def test_run_directory_error_is_isolated(tmp_path, monkeypatch):
    mkdir = Path.mkdir

    def deny_first_fit(path, *args, **kwargs):
        if path.name == "denied":
            raise PermissionError("test per-fit storage error")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_first_fit)
    spec = validate_spec(
        {
            "version": 1,
            "batch_id": "test",
            "max_parallel": 1,
            "runs": [run("denied", "pass"), run("next", "pass")],
        }
    )
    directory = tmp_path / "output"
    assert run_batch(spec, directory, tmp_path / "scratch", "a1", "pod1") == 1
    failed, succeeded = json.loads((directory / "status.json").read_text())["runs"]
    assert (
        failed["state"] == "failed" and failed["error"] == "test per-fit storage error"
    )
    assert succeeded["state"] == "succeeded"


def test_resource_summary_reports_mig_sampling_errors(tmp_path, monkeypatch):
    raw_uuid = "00000000-0000-0000-0000-000000000001"
    info = {
        "cuda_uuid": f"GPU-{raw_uuid}",
        "cuda_visible_devices": "0",
        "nvidia_visible_devices": f"MIG-{raw_uuid}",
        "name": "MIG 1g.10gb",
    }
    assert resources.device_uuids(info)[0] == f"MIG-{raw_uuid}"

    class BrokenDeviceMemory:
        def __init__(self, _info):
            self.uuid = f"MIG-{raw_uuid}"

        def read(self):
            raise RuntimeError("NVML test error")

    monkeypatch.setattr(resources, "DeviceMemory", BrokenDeviceMemory)
    sampler = resources.ResourceSampler(tmp_path, tmp_path, info)
    sampler.sample()
    summary = sampler.summary()
    assert summary["samples"] == 1 and summary["cuda_samples"] == 0
    assert summary["cuda_error"] == "NVML test error"


def test_resource_sampler_resolves_mig_under_the_cuda_parent(monkeypatch):
    raw_uuid = "00000000-0000-0000-0000-000000000001"
    mig_uuid = "MIG-00000000-0000-0000-0000-000000000002"
    other_parent = "GPU-00000000-0000-0000-0000-000000000003"
    other_mig = "MIG-00000000-0000-0000-0000-000000000004"
    listing = (
        f"GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-{raw_uuid})\n"
        f"  MIG 1g.10gb     Device  0: (UUID: {mig_uuid})\n"
        f"GPU 1: NVIDIA A100 80GB PCIe (UUID: {other_parent})\n"
        f"  MIG 1g.10gb     Device  0: (UUID: {other_mig})\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: listing)
    requested = []

    def get_handle(uuid, _handle):
        requested.append(uuid.decode())
        return 0 if uuid.decode() in (f"GPU-{raw_uuid}", mig_uuid) else 6

    library = SimpleNamespace(
        nvmlInit_v2=lambda: 0,
        nvmlDeviceGetHandleByUUID=get_handle,
        nvmlShutdown=lambda: 0,
    )
    monkeypatch.setattr(resources.ctypes, "CDLL", lambda _name: library)
    info = {
        "cuda_uuid": raw_uuid,
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "void",
        "name": "NVIDIA A100 80GB PCIe MIG 1g.10gb",
    }
    device = resources.DeviceMemory(info)
    assert device.uuid == mig_uuid
    assert requested == [f"MIG-{raw_uuid}", mig_uuid]


def test_mig_child_requires_a_unique_match_on_the_cuda_parent():
    parent = "GPU-00000000-0000-0000-0000-000000000001"
    listing = (
        f"GPU 0: NVIDIA A100 80GB PCIe (UUID: {parent})\n"
        "  MIG 1g.10gb Device 0: (UUID: MIG-first)\n"
        "  MIG 1g.10gb Device 1: (UUID: MIG-second)\n"
    )
    assert resources.mig_child_uuid(listing, parent) is None


def test_term_stops_queue_and_kills_a_stubborn_grandchild(tmp_path):
    ready = tmp_path / "ready"
    child_done = tmp_path / "child-done"
    grandchild_code = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).touch()
time.sleep(4.5)
Path(sys.argv[2]).touch()
"""
    code = """
import os, subprocess, sys
with open(os.devnull, 'wb') as sink:
    subprocess.Popen([sys.executable, '-c', *sys.argv[1:]], stdout=sink)
    while True:
        __import__('time').sleep(.1)
"""
    args = command(
        tmp_path,
        [
            run("running", code, grandchild_code, str(ready), str(child_done)),
            run("queued", "raise SystemExit(99)"),
        ],
        max_parallel=1,
    )
    runner = subprocess.Popen(args)
    try:
        wait_for(ready)
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=8) == 128 + signal.SIGTERM
        records = json.loads((output(tmp_path) / "status.json").read_text())["runs"]
        assert [r["state"] for r in records] == ["interrupted", "not_started"]
        time.sleep(2)
        assert not child_done.exists()
    finally:
        if runner.poll() is None:
            os.kill(runner.pid, signal.SIGKILL)
            runner.wait(timeout=5)
