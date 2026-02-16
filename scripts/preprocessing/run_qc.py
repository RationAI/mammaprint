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
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync",
        "uv run preprocessing/quality_control.py data=???",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
