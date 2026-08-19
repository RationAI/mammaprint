from humanfriendly import parse_size
from kube_jobs import storage, submit_job


# Pod resources. These are the single source of truth: they size the K8s request/limit AND
# are forwarded to Ray via environment variables so tiling.py does not hard-code them.
CPU = 48
MEMORY = "64Gi"
SHM = "32Gi"

USERNAME = "kissmi"
GIT_BRANCH = "feat/tiling-values"
EXPERIMENT = "preprocessing/tiling/cancer_mask/mou_3_224"
DATA_NAME = "mou_3_224_cancer_probability_threshold_scan"

# Ray's auto-detection is unreliable on this cluster (it over-detects CPUs from the node and
# under-sizes the object store), so we hand it the pod's real numbers. tiling.py reads these
# and derives ray.init(num_cpus=..., object_store_memory=fraction * RAY_MEMORY_BYTES).
ray_env = [
    f"export RAY_NUM_CPUS={CPU}",
    f"export RAY_MEMORY_BYTES={parse_size(MEMORY, binary=True)}",
]

submit_job(
    job_name="mammaprint-tiling-cancer-probability-l3-threshold-diagnostics",
    username=USERNAME,
    image="cerit.io/rationai/base:2.0.6",
    cpu=CPU,
    memory=MEMORY,
    shm=SHM,
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        f"git checkout {GIT_BRANCH}",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        *ray_env,
        "uv sync --frozen",
        (
            f"uv run -m preprocessing.tiling +experiment={EXPERIMENT} "
            "cancer_threshold=null diagnostics=true "
            f"data_name={DATA_NAME}"
        ),
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
