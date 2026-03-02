from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-preprocessing-epithelium-masks",
    username=...,
    cpu=1,
    memory="4Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.epithelium_masks",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
