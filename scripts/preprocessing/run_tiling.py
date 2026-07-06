from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-tiling-...",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=48,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv sync --frozen",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
