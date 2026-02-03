from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-tiling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run preprocessing/tiling.py",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
