"""Log this user's training jobs, retry the most-recently-failed one, wait, repeat.

The jobs submitted by ``run_train_sweep*.py`` set ``backoff_limit=0`` — one pod
failure marks the whole Job ``Failed`` with no automatic retry. This helper:

  1. lists your jobs (label ``created_by=<USER>``) with their status,
  2. finds the most recent FAILED job,
  3. re-creates it under a bumped name (its own manifest, so image / resources /
     command are preserved exactly — works for any job, transformer or spatial),
  4. sleeps ``WAIT_MINUTES``, then loops.

A failed job whose name ends in ``-<n>`` is retried as ``-<n+1>``; otherwise ``-2``
is appended. The old failed Job is deleted first (its name is a unique key, so a
re-create with the same name would 409). Kubernetes' own TTL may already have
removed it — that's fine.

Usage:
    uv run python scripts/ml/retry_failed_jobs.py                # loop forever, 30 min
    uv run python scripts/ml/retry_failed_jobs.py --once         # one pass, no wait
    uv run python scripts/ml/retry_failed_jobs.py --wait 30      # 30-minute cadence
    uv run python scripts/ml/retry_failed_jobs.py --dry-run      # log + show plan, no changes

Reads/writes only jobs labelled with your username. Requires a working kubectl
context (the same one kube_jobs submits to).
"""

import argparse
import copy
import re
import time

from kube_jobs.constants import CLUSTER, JOB_NAMESPACE
from kubernetes import client
from kubernetes.config import load_kube_config


USER = "kissmi"  # created_by label — the jobs this script is allowed to touch
WAIT_MINUTES = 30
# Only consider these when picking "the last failed job", so we never resurrect an
# unrelated failure (e.g. a preprocessing job). Narrow this if you want.
NAME_PREFIXES = (
    "mammaprint-train-",
    "mammaprint-sweep-",
    "mp-doe36-",
    "mp-doe48-",
)


def _batch_api() -> client.BatchV1Api:
    load_kube_config(context=CLUSTER)
    return client.BatchV1Api()


def _is_failed(job: client.V1Job) -> bool:
    return bool(job.status and job.status.failed)


def _is_active(job: client.V1Job) -> bool:
    return bool(job.status and job.status.active)


def _is_succeeded(job: client.V1Job) -> bool:
    return bool(job.status and job.status.succeeded)


def _status_str(job: client.V1Job) -> str:
    if _is_active(job):
        return "ACTIVE"
    if _is_succeeded(job):
        return "SUCCEEDED"
    if _is_failed(job):
        return "FAILED"
    return "UNKNOWN"


def list_jobs(batch: client.BatchV1Api) -> list[client.V1Job]:
    """This user's jobs, newest first (by start time)."""
    jobs = batch.list_namespaced_job(
        namespace=JOB_NAMESPACE, label_selector=f"created_by={USER}"
    ).items

    def start_key(j: client.V1Job) -> str:
        st = j.status.start_time if j.status else None
        return st.isoformat() if st else ""

    return sorted(jobs, key=start_key, reverse=True)


def log_jobs(jobs: list[client.V1Job]) -> None:
    print(f"\n{'STATUS':<10} {'START':<22} NAME")
    print("-" * 90)
    for j in jobs:
        st = j.status.start_time if j.status else None
        start = st.strftime("%Y-%m-%d %H:%M:%S") if st else "-"
        print(f"{_status_str(j):<10} {start:<22} {j.metadata.name}")


def _bump_name(name: str) -> str:
    """`...-bce-1` -> `...-bce-2`; a name without a trailing `-<n>` gets `-2`."""
    m = re.match(r"^(.*)-(\d+)$", name)
    if m:
        return f"{m.group(1)}-{int(m.group(2)) + 1}"
    return f"{name}-2"


def _retry_manifest(job: client.V1Job, new_name: str) -> client.V1Job:
    """Copy a Job's spec into a fresh Job named `new_name`, stripping runtime fields.

    Kubernetes injects immutable fields into a running Job's pod template (the
    `controller-uid`/`job-name` selector labels and `batch.kubernetes.io/*`). Those
    must be removed or the create is rejected — we rebuild a clean metadata/spec and
    carry over only what we set at submit time.
    """
    src = job.spec
    tmpl = copy.deepcopy(src.template)

    # Drop the auto-injected selector labels; keep our own app/created_by labels.
    labels = dict(tmpl.metadata.labels or {}) if tmpl.metadata else {}
    for k in list(labels):
        if k.startswith("batch.kubernetes.io/") or k in {"controller-uid", "job-name"}:
            del labels[k]
    labels["app"] = new_name
    tmpl.metadata = client.V1ObjectMeta(labels=labels)

    # Rename the container so its name matches the new job (cosmetic, matches submit).
    for c in tmpl.spec.containers:
        c.name = new_name

    new_spec = client.V1JobSpec(
        backoff_limit=src.backoff_limit,
        ttl_seconds_after_finished=src.ttl_seconds_after_finished,
        template=tmpl,
    )
    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=new_name, labels={"created_by": USER}),
        spec=new_spec,
    )


def last_failed(jobs: list[client.V1Job]) -> client.V1Job | None:
    for j in jobs:
        if j.metadata.name.startswith(NAME_PREFIXES) and _is_failed(j):
            return j
    return None


def retry_last_failed(
    batch: client.BatchV1Api, jobs: list[client.V1Job], dry_run: bool
) -> str | None:
    """Delete the newest failed job and re-create it under a bumped name.

    Returns the new job name, or None if there was nothing to retry.
    """
    target = last_failed(jobs)
    if target is None:
        print("\nNo failed job to retry. ✅")
        return None

    old = target.metadata.name
    new = _bump_name(old)
    print(f"\nLast failed job: {old}  ->  retry as: {new}")

    if dry_run:
        print(
            "[dry-run] would delete the old job and create the retry; no changes made."
        )
        return new

    manifest = _retry_manifest(target, new)

    # Delete the old failed Job so its name is free (may already be TTL-collected).
    try:
        batch.delete_namespaced_job(
            name=old, namespace=JOB_NAMESPACE, propagation_policy="Background"
        )
        print(f"Deleted old failed job {old}.")
    except client.ApiException as e:
        if e.status == 404:
            print(f"Old job {old} already gone (TTL); nothing to delete.")
        else:
            raise

    try:
        resp = batch.create_namespaced_job(body=manifest, namespace=JOB_NAMESPACE)
        print(f"Retry submitted: {resp.metadata.name} 🚀")
        return resp.metadata.name
    except client.ApiException as e:
        print(f"Retry creation failed: {e.reason} ({e.status})")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_MINUTES,
        help=f"Minutes to wait between passes (default {WAIT_MINUTES}).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass (log + retry) and exit, no waiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log jobs and show the retry plan without changing anything.",
    )
    args = parser.parse_args()

    batch = _batch_api()
    pass_num = 0
    while True:
        pass_num += 1
        print(f"\n{'=' * 90}\nPASS {pass_num}\n{'=' * 90}")
        jobs = list_jobs(batch)
        log_jobs(jobs)
        retry_last_failed(batch, jobs, dry_run=args.dry_run)

        if args.once:
            break

        wait_s = args.wait * 60
        print(f"\nSleeping {args.wait:g} min before the next pass (Ctrl-C to stop)...")
        try:
            time.sleep(wait_s)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
