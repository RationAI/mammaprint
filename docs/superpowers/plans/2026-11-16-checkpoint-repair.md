# Checkpoint Upload Repair and Bounded Recovery Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Every logical change and every review fix requires independent review before its focused commit.

**Goal:** Keep uploaded checkpoints current and prepare a bounded, provenance-preserving retraining workflow from an existing MLflow run link.

**Architecture:** A project-local logger subclass fixes change detection without modifying the pinned dependency. A recovery module plans from archived config and complete validation history, reuses native training with a stopping callback, and publishes into a linked recovery run. A separate guarded operation can replace the original best artifact after verified backups. A thin cluster launcher reuses existing job conventions.

**Spec:** User requests 2026-11-16: fix checkpoint uploading; prepare a script using an existing MLflow link, same hyperparameters/seed, stop at the historical validation optimum, and replace the model in the same experiment. Preparation does not authorize actual retraining, cluster submission or remote artifact replacement.

## Global constraints and decisions

- Preserve original model/data/optimizer/scheduler/seed and training settings. Do not edit installed/shared .venv packages or relax scientific source compatibility.
- Default selection follows the original ModelCheckpoint monitor/mode (audited runs: valid/loss, min). Explicit validation-only monitor/mode overrides are marked changed selection. Never select using test scores.
- Same seed is not a guarantee of identical lost weights. Record source/config/code/environment/data identity and limits; describe results as retraining, not byte-identical reconstruction.
- Start from original seeded initialization, not the stale epoch-0 checkpoint without original RNG state.
- Keep original trainer.max_epochs/max_steps and scheduler horizon. Stop after the recorded best epoch via callback; never run the original patience tail. Optional stop-on-target cannot exceed this cap. Refuse automatic publication on missed/nonfinite target or incomplete provenance.
- Default metadata planning is read-only and bounded; --help and cluster dry-run remain stdlib/offline. Explicit --execute trains only in a new linked run in the SAME experiment. No post-fit test by default and no copying historical test metrics.
- Publication is separate: explicit --publish --replace-original plus successful recovery manifest and expected-old SHA. Preserve/read-verify old weights and MLmodel, stage/read-verify new bytes, recheck original, replace only canonical best payload/manifest, then read-verify. Preserve source config/console/metric histories and tag test metrics as not revalidated.
- MLflow multi-object overwrite is non-atomic; serialize repairs per source, retain a journal/backups, and report partial failures. No broad deletion or automatic rollback pretending transactional safety.
- Use exact fixed implementation SHA and historical source SHA separately. Historical config must explicitly select the repaired logger. Existing training launchers on old refs do not receive the fix automatically.
- No real training, test scoring, cluster calls/submissions, artifact writes or source-run changes during implementation/verification. Local synthetic/fake-client tests only; authorized read-only cached metadata may validate plans.

## Task 1: Correct checkpoint uploads

Files: new ml/loggers.py, configs/logger/mlflow.yaml, minimal ml/train.py annotation import, tests/test_mlflow_logger.py.

- Reproduce same-path best.ckpt replacement being skipped by the installed logger with a local fake artifact client.
- Subclass the existing logger; retain its byte-copy/MLmodel helper and pruning contract. Track successful per-run/per-stem upload fingerprints and content hashes; changed best/last overwrite, unchanged bytes skip, failed uploads retry. A fresh logger must not trust a pre-existing remote stem.
- Preserve rank-zero behavior, disabled/True/all timing and final flush, including last symlinks. Never mark failed uploads successful.
- Native tiny CPU Trainer regression forces multiple best improvements and distinct last; assert remote copied bytes/weights/epoch match final local best, plus mode/rank/failure cases.
- Focused lint/tests, independent review/fix/re-review, focused commit.

## Task 2: Recovery planning, execution and guarded publication

Files: new ml/recover_checkpoint.py, tests/test_checkpoint_recovery.py, focused recovery documentation as needed. Logger class is supplied by Task 1.

Public CLI contract: python -m ml.recover_checkpoint SOURCE --tracking-uri URI --implementation-sha SHA --output-root ROOT; default prints metadata plan. --execute trains to the bounded plan. Separate --publish --replace-original --manifest PATH --expected-old-sha256 SHA performs publication only. Execution and publication are mutually exclusive. Source URL experiment/server/run must match explicit identity. Optional --monitor and --mode must be supplied together and validation-only.

- Pure URL/config/history planning first: parse original resolved bytes, status/experiment/source SHA, complete bounded monitor and epoch history; choose earliest eligible optimum, align exact epoch, reject ambiguous/truncated/nonfinite histories and unsupported validation cadence. Never infer best from latest metric values.
- Preserve original horizon/minimum bounds; reject a cap that Lightning minimums would ignore. Verify training-critical source and dependency pins, permitting unrelated explainability extras while recording unverified original runtime/data bytes.
- Implement a small callback and native seed/instantiate/fit path. Create one linked recovery run, use the fixed logger, preserve effective-config allowlist diff, fit train/val only, validate best metadata/hash and target outcome, read-verify recovery artifact. No test call.
- Native tiny CPU tests prove original scheduler/LR prefix and checkpoint-save semantics survive early cap, epoch0 means one completed epoch, target miss blocks replacement, source configs remain unchanged.
- Fake-client publication tests prove expected-old guard before writes, immutable backup/stage readbacks, no root config/console/history modifications, final byte/MLmodel verification, and explicit partial-failure journal. No old test metric copying.
- Focused tests/lint and independent review/fixes/re-review, focused commit with matching workflow skill in Task 3 if one shared logical deliverable.

## Task 3: Cluster launcher and reusable recovery skill

Files: scripts/ml/run_checkpoint_recovery.py, tests/test_run_checkpoint_recovery.py, .agents/skills/mammaprint-checkpoint-recovery/SKILL.md.

- Reuse established helper target, secure mounts, secret references and concept launcher's validated resource/submission utilities without changing concept behavior. Default prints an offline plan; only --submit creates a job, never publication.
- Exact implementation checkout, historical object availability, frozen environment and CUDA/native preflight; invoke recovery --execute in one job. No --publish in generated training jobs.
- Document bounded metadata planning, objective/epoch cap, same-experiment linked run, separate publication guards/backup/readback, non-atomic failure/rollback, monitoring and prerequisites. Actual submit and replacement need explicit authorization.
- Offline isolation/quoting/config/resource/fake submit tests; skill checks; independent review and focused commit with script+skill.

## Final verification

- Whole local suite and final CLI/launcher integration; independently reviewed synthetic recovery lifecycle (no remote mutations).
- Read-only plan against existing cached run evidence where feasible; confirm expected target epochs without training.
- Whole-change independent review, resolve findings, focused commits and push the completed branch under existing user authorization. No training, cluster submission or artifact replacement.
