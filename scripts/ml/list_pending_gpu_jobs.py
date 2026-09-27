"""List this user's Kubernetes jobs that are waiting for a GPU.

The GPU request is read from the pod spec created by ``kube_jobs.submit_job``.
This script is read-only.

Usage:
    uv run python scripts/ml/list_pending_gpu_jobs.py
    uv run python scripts/ml/list_pending_gpu_jobs.py --prefix mammaprint-sweep-
    uv run python scripts/ml/list_pending_gpu_jobs.py --user kissmi
"""

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping

from kube_jobs.constants import CLUSTER, JOB_NAMESPACE
from kubernetes import client
from kubernetes.config import load_kube_config


DEFAULT_USER = "kissmi"
GPU_PRODUCTS = {
    "NVIDIA-A100-80GB-PCIe": "A100",
    "NVIDIA-A40": "A40",
    "NVIDIA-H100-NVL": "H100",
    "Tesla-P100-SXM2-16GB": "Tesla P100",
}


def balanced_gpu_iterator(waiting_counts: Mapping[str, int]) -> Iterator[str]:
    """Yield GPUs forever, first bringing their waiting counts to the same level.

    Ties follow the mapping's insertion order. The input mapping is not modified.
    Include every GPU that may be selected, using a count of zero when its queue is
    empty.
    """
    if not waiting_counts:
        raise ValueError("At least one GPU is required.")
    if any(count < 0 for count in waiting_counts.values()):
        raise ValueError("Waiting counts cannot be negative.")

    counts = dict(waiting_counts)
    while True:
        lowest_count = min(counts.values())
        for gpu in counts:
            if counts[gpu] == lowest_count:
                counts[gpu] += 1
                yield gpu


def _gpu_name(pod: client.V1Pod) -> str:
    limits = {
        resource: quantity
        for container in pod.spec.containers
        for resource, quantity in (container.resources.limits or {}).items()
    }

    for resource in sorted(limits):
        if resource.startswith("nvidia.com/mig-"):
            return resource.removeprefix("nvidia.com/")

    if "nvidia.com/gpu" in limits:
        selectors = pod.spec.node_selector or {}
        product = selectors.get("nvidia.com/gpu.product", "GPU")
        return GPU_PRODUCTS.get(product, product)

    return "none"


def _pending_gpu_jobs(user: str, prefix: str | None) -> list[tuple[str, str]]:
    load_kube_config(context=CLUSTER)
    jobs = (
        client.BatchV1Api()
        .list_namespaced_job(
            namespace=JOB_NAMESPACE,
            label_selector=f"created_by={user}",
        )
        .items
    )
    job_names_by_uid = {
        job.metadata.uid: job.metadata.name
        for job in jobs
        if job.metadata is not None
        and job.metadata.uid is not None
        and (prefix is None or job.metadata.name.startswith(prefix))
    }

    pending = []
    pods = client.CoreV1Api().list_namespaced_pod(namespace=JOB_NAMESPACE).items
    for pod in pods:
        if pod.status is None or pod.status.phase != "Pending":
            continue
        owner_uids = {
            owner.uid for owner in (pod.metadata.owner_references or []) if owner.uid
        }
        job_uid = next((uid for uid in owner_uids if uid in job_names_by_uid), None)
        if job_uid is not None:
            pending.append((_gpu_name(pod), job_names_by_uid[job_uid]))

    return sorted(pending)


def pending_gpu_counts(user: str, prefix: str | None = None) -> Counter[str]:
    return Counter(gpu for gpu, _ in _pending_gpu_jobs(user, prefix))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help=f"Value of the created_by job label (default: {DEFAULT_USER}).",
    )
    parser.add_argument(
        "--prefix",
        help="Only show job names starting with this prefix.",
    )
    args = parser.parse_args()

    pending = _pending_gpu_jobs(args.user, args.prefix)
    if not pending:
        print("No pending jobs found.")
        return 0

    counts = Counter(gpu for gpu, _ in pending)
    print(
        "Waiting by GPU: "
        + ", ".join(f"{gpu}={count}" for gpu, count in sorted(counts.items()))
    )
    print(f"\n{'GPU':<16} JOB")
    print("-" * 80)
    for gpu, job_name in pending:
        print(f"{gpu:<16} {job_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
