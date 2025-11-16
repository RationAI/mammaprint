from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-preprocessing-qc",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="8Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync",
        "uv run preprocessing/quality_control.py",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
