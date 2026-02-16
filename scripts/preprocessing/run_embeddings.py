from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-embeddings-1",
    username="550439",
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "git checkout feature/embeddings",
        "export HF_TOKEN=<YOUR_HF_TOKEN>",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings data/tiled=0_224",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
