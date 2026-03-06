from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-tiling-...",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,
    memory="64Gi",
    shm="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
