"""Render one seed-0 multiscale replay Job without contacting Kubernetes."""

import argparse
import json
import re


RETRIES = {
    "cm5": ("a24468e200d24c569e9e68c06107b70c", 230),
    "ean5": ("6405fef8a83e4e7e985d3dd34738ee00", 200),
}
REPOSITORY = "https://github.com/RationAI/mammaprint.git"
IMAGE = "cerit.io/rationai/base:2.0.6"
SCRATCH = "/replay-scratch"
MAPPING = "/mnt/projects/mammaprint/data_mapping.csv"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def render_job(args: argparse.Namespace) -> dict:
    source_run, default_scratch = RETRIES[args.dataset]
    name = f"mp-ms-replay-{args.dataset}-slide-attention-s0-{args.source_commit[:8]}"
    scratch = f"{args.scratch_gib or default_scratch}Gi"
    resources = {
        "cpu": str(args.cpu),
        "memory": f"{args.memory_gib}Gi",
        "ephemeral-storage": scratch,
        "nvidia.com/gpu": "1",
    }
    worker = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "ml.train",
        f"+experiment=ml/multiscale/screen/{args.dataset}_slide_attention_regression",
        "seed=0",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "+logger.tags.multiscale_screen=slide_attention_regression",
        f"+logger.tags.data_variant={args.dataset}",
        "+logger.tags.sweep_seed=seed_0",
        "test_after_fit=true",
        f"metadata.run_name={name}",
        f"+logger.tags.recovery_source_run_id={source_run}",
    ]
    bootstrap = "\n".join(
        [
            "set -u -o pipefail",
            "umask 077",
            "unset MLFLOW_RUN_ID MLFLOW_PARENT_RUN_ID",
            f"test -r {MAPPING}",
            'mkdir -p "$TMPDIR" "$UV_CACHE_DIR"',
            "export GIT_CONFIG_COUNT=1",
            "export GIT_CONFIG_KEY_0=credential.https://gitlab.ics.muni.cz.helper",
            'export GIT_CONFIG_VALUE_0=\'!f() { if [ "$1" = get ]; then '
            'printf "%s\\n" "username=$VCS_USER" "password=$VCS_PASSWORD"; fi; }; f\'',
            f"git clone --no-checkout {REPOSITORY} workdir",
            'git -C workdir checkout --detach "$1"',
            'git -C workdir rev-parse HEAD | grep -Fx "$1"',
            "shift",
            "cd workdir",
            "uv sync --frozen",
            'exec "$@"',
        ]
    )
    env_values = {
        "MLFLOW_TRACKING_URI": args.tracking_uri,
        "MLFLOW_TRACKING_USERNAME": args.username,
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "600",
        "HTTPS_PROXY": "http://proxy.ics.muni.cz:3128",
        "NO_PROXY": "mlflow-s3.rationai-mlflow,mlflow.rationai-mlflow,svc.cluster.local",
        "HYDRA_FULL_ERROR": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "TMPDIR": f"{SCRATCH}/tmp",
        "UV_CACHE_DIR": f"{SCRATCH}/uv-cache",
    }
    env = [{"name": key, "value": value} for key, value in env_values.items()]
    for name_key, secret_key in (
        ("VCS_USER", "username"),
        ("VCS_PASSWORD", "password"),
    ):
        env.append(
            {
                "name": name_key,
                "valueFrom": {
                    "secretKeyRef": {"name": args.gitlab_secret, "key": secret_key}
                },
            }
        )
    container = {
        "name": "train",
        "image": IMAGE,
        "imagePullPolicy": "Always",
        "workingDir": SCRATCH,
        "command": ["/bin/bash", "-ec"],
        "args": [bootstrap, "multiscale-retry", args.source_commit, *worker],
        "env": env,
        "resources": {"requests": resources, "limits": resources},
        "volumeMounts": [
            {"name": "projects", "mountPath": "/mnt/projects", "readOnly": True},
            {"name": "scratch", "mountPath": SCRATCH},
        ],
        "securityContext": {
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": args.namespace,
            "labels": {"created_by": args.username},
            "annotations": {
                "mammaprint.rationai.org/target-context": args.context,
                "mammaprint.rationai.org/source-commit": args.source_commit,
                "mammaprint.rationai.org/recovery-source-run": source_run,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "nodeSelector": {"nvidia.com/gpu.product": "NVIDIA-A40"},
                    "securityContext": {
                        "runAsNonRoot": True,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": [
                        {
                            "name": "projects",
                            "persistentVolumeClaim": {
                                "claimName": args.projects_claim,
                                "readOnly": True,
                            },
                        },
                        {"name": "scratch", "emptyDir": {"sizeLimit": scratch}},
                    ],
                    "containers": [container],
                }
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=RETRIES, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--tracking-uri", default="http://mlflow-s3.rationai-mlflow")
    parser.add_argument("--projects-claim", default="projects-rw")
    parser.add_argument("--gitlab-secret", default="gitlab-deploy-token")
    parser.add_argument("--cpu", type=positive_int, default=4)
    parser.add_argument("--memory-gib", type=positive_int, default=16)
    parser.add_argument("--scratch-gib", type=positive_int)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_commit):
        parser.error("--source-commit must be a full lowercase Git SHA")
    print(json.dumps(render_job(args), indent=2))


if __name__ == "__main__":
    main()
