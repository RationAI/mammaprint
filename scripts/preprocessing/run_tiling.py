from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-tiling-...",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "git checkout feature/experiment-refactor",
        "uv sync --frozen",
        "uv run preprocessing/tiling.py experiment/preprocessing/tiling=???",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
