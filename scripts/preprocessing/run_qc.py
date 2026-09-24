from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-preprocessing-qc",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv sync --frozen",
        "uv run -m preprocessing.quality_control",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
