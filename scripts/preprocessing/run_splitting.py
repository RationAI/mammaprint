from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-splitting-...",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="8Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.splitting source_dataset_uri=??? data_name=???",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
