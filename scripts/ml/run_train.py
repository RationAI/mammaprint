from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-train-mil-embeddings",
    username="kissmi",
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        # "git checkout <branch>",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "export HF_TOKEN=",
        "uv sync --frozen",
        "uv run -m ml.train +experiment=ml/train_mil_embeddings",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
