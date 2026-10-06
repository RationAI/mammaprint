"""Submit the prefilled cancer-threshold analysis to the cluster."""

from kube_jobs import submit_job


submit_job(
    job_name="mammaprint-cancer-probability-threshold-analysis-l3",
    username="kissmi",
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/rationAI/mammaprint workdir",
        "cd workdir",
        "git checkout feat/tiling-values",
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "uv sync --frozen",
        "uv run python scripts/notebooks/mask_coverage_threshold.py",
    ],
)
