from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-preprocessing-epithelium-masks",
    username=...,
    cpu=1,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv sync --frozen",
        "uv run -m preprocessing.epithelium_masks",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
