"""Re-run the test-set inference for a run whose post-fit test step died, and append
the results to that **same MLflow run**.

Background: ``ml/train.py`` does ``fit`` then, if ``test_after_fit``, ``test`` on the
best checkpoint. When a slide download fails during that test step, the run ends up
with a fully trained model (its best checkpoint is logged under the run's
``checkpoints/best`` artifacts) and train/val metrics — but **no test metrics**. There
is no need to retrain; we only need to re-run inference and write it back.

This entrypoint, given a run id (or run name), does exactly that:

  1. reads the run's own logged Hydra overrides (``configs/hydra.yaml`` artifact),
     so the data shape / model pieces match the original run exactly,
  2. downloads the run's best checkpoint (``checkpoints/best/checkpoint.ckpt``),
  3. re-invokes ``ml.train`` in ``mode=test`` with ``checkpoint=<that ckpt>`` and
     ``+logger.run_id=<run id>`` — Lightning's MLFlowLogger resumes the existing run
     when ``run_id`` is set, so the new ``test/*`` metrics append to it.

Console log: mlkit's StreamCapture re-logs the run's whole console to a single
``console.log`` artifact from an empty buffer, so the resume's ``test`` output would
overwrite the original training log. Before launching, we copy the existing
``console.log`` to ``console.train.log`` to preserve it (disable with
``--no-keep-console-log``).

Must run in an environment with the data mounts and MLflow reachable — i.e. the same
pod shape the training job used (see ``scripts/ml/submit_resume_test.py``).

Usage (inside such a pod):
    uv run -m ml.resume_test --run-id <mlflow-run-id>
    uv run -m ml.resume_test --run-name <job/run-name> --experiment MammaPrint
    uv run -m ml.resume_test --run-id <id> --dry-run     # print the command, run nothing
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from mlflow import MlflowClient
from omegaconf import OmegaConf


DEFAULT_EXPERIMENT = "MammaPrint"

# Artifact paths written by rationai.mlkit (autolog._log_config / MLFlowLogger).
HYDRA_ARTIFACT = "configs/hydra.yaml"  # holds hydra.overrides.task (the CLI args)
BEST_CKPT_ARTIFACT = "checkpoints/best/checkpoint.ckpt"

# mlkit's StreamCapture re-logs the run's whole console to this single path, from an
# empty buffer — so the resume's `test` output would OVERWRITE the training console
# log. We back the original up under this path before launching the subprocess.
CONSOLE_LOG = "console.log"
CONSOLE_LOG_BACKUP = "console.train.log"

# Overrides we must drop from the replay: they steer stage/checkpoint/run identity,
# which resume_test sets itself. Anything matching `key=...` (with these keys) is cut.
DROP_OVERRIDE_KEYS = {"mode", "checkpoint", "test_after_fit"}


def _client(tracking_uri: str | None) -> MlflowClient:
    # tracking_uri=None -> MlflowClient reads MLFLOW_TRACKING_URI from the env.
    return MlflowClient(tracking_uri=tracking_uri)


def resolve_run_id(
    mlflow_client: MlflowClient, run_id: str | None, run_name: str | None, experiment: str
) -> str:
    """Return the run id, looking it up by run name in `experiment` if needed."""
    if run_id:
        return run_id
    exp = mlflow_client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"MLflow experiment {experiment!r} not found.")
    safe = run_name.replace("'", "\\'")
    runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.run_name = '{safe}'",
    )
    if not runs:
        raise SystemExit(f"No MLflow run named {run_name!r} in {experiment!r}.")
    if len(runs) > 1:
        ids = ", ".join(r.info.run_id for r in runs)
        raise SystemExit(
            f"{len(runs)} runs named {run_name!r} in {experiment!r} ({ids}); "
            f"pass --run-id to disambiguate."
        )
    return runs[0].info.run_id


def original_overrides(mlflow_client: MlflowClient, run_id: str, dst: Path) -> list[str]:
    """Read the run's original Hydra CLI overrides, minus the stage/checkpoint ones.

    ``autolog._log_config`` saves ``hydra.yaml`` via ``OmegaConf.save(HydraConfig.get())``.
    ``HydraConfig.get()`` returns the *contents* of the ``hydra`` node, so the CLI args
    live at ``overrides.task`` (no ``hydra.`` prefix). We try both keys to be safe.
    """
    path = mlflow_client.download_artifacts(run_id, HYDRA_ARTIFACT, str(dst))
    hydra_cfg = OmegaConf.load(path)
    task = OmegaConf.select(hydra_cfg, "overrides.task")
    if task is None:
        task = OmegaConf.select(hydra_cfg, "hydra.overrides.task")
    task = list(task or [])
    if not task:
        raise SystemExit(
            f"Run {run_id}: no Hydra overrides found in {HYDRA_ARTIFACT} "
            f"(looked at overrides.task and hydra.overrides.task). Without them the "
            f"experiment/data config can't be reconstructed. Pass the overrides "
            f"manually, or inspect the artifact."
        )

    kept = []
    for ov in task:
        key = ov.split("=", 1)[0].lstrip("+~")
        if key in DROP_OVERRIDE_KEYS:
            continue
        kept.append(ov)
    return kept


def download_checkpoint(mlflow_client: MlflowClient, run_id: str, dst: Path) -> str:
    """Download the run's best checkpoint; return its local path."""
    # Fail early with a clear message if the fit never logged a checkpoint.
    arts = {a.path for a in mlflow_client.list_artifacts(run_id, "checkpoints/best")}
    if BEST_CKPT_ARTIFACT not in arts:
        raise SystemExit(
            f"Run {run_id} has no {BEST_CKPT_ARTIFACT!r} artifact "
            f"(found: {sorted(arts) or 'nothing'}). Cannot resume test without a "
            f"trained checkpoint — the fit likely did not complete."
        )
    return mlflow_client.download_artifacts(run_id, BEST_CKPT_ARTIFACT, str(dst))


def backup_console_log(
    mlflow_client: MlflowClient, run_id: str, dst: Path, dry_run: bool
) -> bool:
    """Preserve the original training console log before the resume overwrites it.

    The resume subprocess writes a fresh ``console.log`` (mlkit's StreamCapture starts
    from an empty buffer and log_text overwrites), which would clobber the training
    console output. We copy the existing ``console.log`` to ``console.train.log`` first.
    Idempotent-ish: if a backup already exists we don't re-copy (so re-running the
    resume doesn't overwrite the true original with a test-only log). Returns True if a
    backup was made (or already present), False if there was no original to save.
    """
    top = {a.path for a in mlflow_client.list_artifacts(run_id)}
    if CONSOLE_LOG_BACKUP in top:
        print(f"  {CONSOLE_LOG_BACKUP!r} already exists; leaving it as the original.")
        return True
    if CONSOLE_LOG not in top:
        print(f"  No {CONSOLE_LOG!r} to preserve (nothing logged yet).")
        return False

    if dry_run:
        print(f"  [dry-run] would copy {CONSOLE_LOG!r} -> {CONSOLE_LOG_BACKUP!r}.")
        return True

    local = mlflow_client.download_artifacts(run_id, CONSOLE_LOG, str(dst))
    text = Path(local).read_text(encoding="utf-8")
    mlflow_client.log_text(run_id, text, CONSOLE_LOG_BACKUP)
    print(f"  Preserved training log as {CONSOLE_LOG_BACKUP!r}.")
    return True


def build_command(overrides: list[str], checkpoint: str, run_id: str) -> list[str]:
    """The `ml.train` test-mode command that appends results to the existing run."""
    return [
        "uv", "run", "-m", "ml.train",
        *overrides,
        "mode=test",
        f"checkpoint={checkpoint}",
        "test_after_fit=false",
        f"+logger.run_id={run_id}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id", help="MLflow run id to resume test for.")
    src.add_argument("--run-name", help="MLflow run name (== job name) to look up.")
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment to search when using --run-name (default {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI in the env).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra Hydra override to pass to ml.train, repeatable. When given, these "
        "REPLACE the overrides read from the run's logged config (use if the run's "
        "hydra.yaml has none, or to fix them).",
    )
    parser.add_argument(
        "--keep-console-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before running, copy the run's existing console.log to console.train.log "
        "so the resume's test output doesn't clobber the training log (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything and print the command, but don't run test.",
    )
    args = parser.parse_args()

    mlflow_client = _client(args.tracking_uri)
    run_id = resolve_run_id(mlflow_client, args.run_id, args.run_name, args.experiment)
    print(f"Resuming test for run {run_id}.")

    # Keep the downloaded config + checkpoint for the whole subprocess lifetime.
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        if args.override:
            overrides = args.override
            print("Using overrides passed on the command line (skipping logged config).")
        else:
            overrides = original_overrides(mlflow_client, run_id, tmp / "cfg")
        checkpoint = download_checkpoint(mlflow_client, run_id, tmp / "ckpt")
        print(f"Replaying overrides: {overrides}")
        print(f"Checkpoint: {checkpoint}")

        if args.keep_console_log:
            print("Preserving the original console log:")
            backup_console_log(mlflow_client, run_id, tmp / "console", args.dry_run)

        cmd = build_command(overrides, checkpoint, run_id)
        printable = " ".join(cmd)
        if args.dry_run:
            print(f"[dry-run] would run:\n  {printable}")
            return 0

        print(f"Running:\n  {printable}")
        # Inherit stdout/stderr so the test output streams; MLflow also captures it
        # into the same run via the logger's stream capture.
        result = subprocess.run(cmd)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
