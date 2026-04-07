from kube_jobs import storage, submit_job


submit_job(
    job_name="mammaprint-embeddings-...",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/mammaprint workdir",
        "cd workdir",
        "export HF_TOKEN=<YOUR_HF_TOKEN>",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
