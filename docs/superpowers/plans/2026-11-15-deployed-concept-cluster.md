# Deployed Concept Workflow and Cluster Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute each task with independent review before a focused commit.

**Goal:** Reuse Adam's deployed output-metric concept method where compatible with native MammaPrint models, and provide an offline-plannable GPU cluster launcher following existing run scripts.

**Architecture:** Keep the existing copied CPU numerical functions. Add a documented native mean-linear classifier metric adapter and transform-aware dictionary artifacts; add one-process train fit and validation explain with metadata-only split auditing. A separate stdlib launcher generates pinned jobs using the existing `kube_jobs` helper.

**Tech Stack:** Python, NumPy, pandas, PyTorch, Ray, Hydra, MLflow, existing kube_jobs.

**Spec:** The scoped requirements and decisions below implement the user's 2026-11-15 request to use Adam's deployed workflow and create a strong-GPU cluster run script. They supersede the earlier plain semi-NMF replication target; existing plain artifacts remain compatible.

## Global Constraints

- Reuse RationAI/ulcerative-colitis source at `75c22a2a98d3b16a1c298735e2ea385cccfd07fb`; deployed dictionary `be859ae30ca143b78d957493d4f8a8c8`, gallery `b30054c6bffb4df8896c8acb75a48959`. Preserve source/license/provenance and unchanged existing numerical functions.
- No new gradient-based/general-model explainability algorithm, trained-model edits, automatic plain fallback, cluster submission, push, source-run mutation, or live test scoring during implementation.
- Initially support the native IdentityEncoder + MeanPool + single LinearHead + BCE/type graph. Use explicit binary logits `[0,z]` for upstream class-centering semantics. Unsupported heads, index/joint objectives, max/gated-attention/transformer/spatial pooling fail before whole-split downloads.
- Mean pooling has no attention score. Explicitly omit the nonexistent attention term; record adapter/version, requested attention weight1, `attention_applicable=false` and its reason. Do not claim native support by the unchanged UC-specific attention function.
- Deployed profile: K16 configurable; centering true; unscaled signed2560D inputs; one epoch; seed42; batch25600; shuffle250000; NNLS200; forget0.95; semi-NMF ridge1e-6; gauge-fix true; metric ridge0.3 and hash-sample expected20000 tiles. Metric ridge is a separate parameter.
- Preserve exact upstream hash-threshold sampling, transformed fitting, inverse mapping before gauge-fix, transform-aware encoding and raw-space decoding. Sampling is expected-size, not a hard cap.
- Preserve legacy plain dictionary IDs/behavior and original frozen predictions. New metric dictionaries validate/checksum the transform and bind source model/config identity. Never silently drop a transform.
- New `configs/concepts_deployed.yaml` explicitly selects the metric profile; existing plain config/defaults and offline baseline reproduction remain compatible.
- All upstream metric/factorization/NNLS/encoding is CPU. CUDA is used for native PyTorch explanation inference. Do not claim GPU dictionary fitting or rewrite numerical kernels.
- Every logical change and any fixes receive independent review before commit/completion. Coordinator owns commits; no worker recursively delegates.

## Review Focus

- An unsupported source graph must fail before expensive embedding downloads or new result publication.
- Default validation must not download/score test embeddings; metadata-only test membership checks remain allowed and accurately labelled.
- Metric dictionary load must retain its transform and reject cross-model binding mismatches even when embedding corpora match.
- Default launcher planning/help must work with stdlib only, without subprocess/network/cluster calls or filesystem writes.
- Generated jobs must pin implementation and historical training revisions separately, preserve source runs, and verify the actual fixed helper target instead of pretending arbitrary namespace routing.

### Task 1: Metric adapter and dictionary integration

**Files:** new `ml/concept_explainability/metric.py`, new deployed config, changes to `evaluation.py`, `data.py`, narrowly `model.py`/`report.py` if needed, provenance notice, focused concept tests and `docs/concept-explainability.md`.

**Interfaces:** Existing fit/evaluate/report signatures remain compatible. `fit_dictionary` reads optional `fit.metric`; absent/disabled means legacy plain. Expose a small source-graph compatibility check usable before dataset loading. New metric loader supplies validated T to existing `ConceptAutoencoder`.

- [ ] Add failing fixtures for the native classifier metric versus the recovered upstream classifier component, binary centering, invalid spreads, and explicit unsupported-graph rejection.
- [ ] Port only the existing metric algebra/sampling needed for the native classifier-only adaptation, retaining upstream provenance and clearly documenting adaptation. Preserve row/key alignment and ordered Ray streaming.
- [ ] Fit in transformed space and save raw-space atoms plus validated/checksummed transform and source/sample metadata. Use a versioned metric identity; old plain hashes remain unchanged.
- [ ] Test anisotropic transform roundtrip, corruption/missing-transform/binding rejection, raw signed decode, unchanged native weights, legacy plain behavior and train-only sampling.
- [ ] Run focused relevant tests/lint. Independent review, fix/re-review, then focused metric commit.

### Task 2: Efficient one-process cluster workflow

**Files:** `ml/concepts.py`, split-audit helpers in `data.py`, deployed config and focused CLI/data tests.

**Interfaces:** `action=fit_explain`, `output_dir=ROOT`, `include_test=false`; generated subdirectories `ROOT/dictionary`, `ROOT/val`, optional `ROOT/test`. Fit train only, then evaluate/report requested splits using the same `SourceRun` dataset cache and one fresh destination MLflow run. Existing fit/explain actions continue working.

- [ ] Add tests showing metadata-only split audit never instantiates/downloads test datasets, including conservative empty-bag membership and patient/slide leakage rejection.
- [ ] Audit remote flat parquet membership using artifact listings plus original label mapping, without parquet body reads. Retain offline cached fixture support and describe conservative empty-bag handling.
- [ ] Preflight frozen model/profile compatibility before full-split loading. Add combined action and preserve ordering, inherited-run rejection, per-stage immutable outputs and fresh destination-run identity.
- [ ] Test default val-only single-process loading, test opt-in, failure-before-download, and old action compatibility. Independent review and focused commit.

### Task 3: Cluster launcher and reusable skill

**Files:** `scripts/ml/run_concept_explainability.py`, `tests/test_run_concept_explainability.py`, `.agents/skills/mammaprint-concept-cluster/SKILL.md`, focused usage docs.

**Interfaces:** Require explicit source run/training-SHA pairs, implementation SHA, tracking URI and output root. K defaults16. Default plan only; `--submit` explicitly submits. `--context`/`--namespace` are expectations for the existing fixed private helper target (`kubas-cluster`, `rationai-jobs-ns`), not arbitrary routing.

- [ ] Add offline tests for plan isolation, source/implementation pins, resources/quoting, supported-target identity, duplicate rejection and fake submission confirmation.
- [ ] Follow existing clone/uv/image/mount/helper conventions. Default full H100, CPU16, RAM64Gi (configurable); preserve existing secure secret references. Generate fail-fast shell, exact detached checkout, historical-object fetch, explainability extras, CUDA/native-library preflight, unset inherited `MLFLOW_RUN_ID`, and one combined deployed action per source/K.
- [ ] Confirm named job creation because existing helper swallows create errors. No live submission or cluster inspection in local tests.
- [ ] Skill documents prerequisites/auth references without secrets, exact script/plan/submit commands, identity checks, source reachability, expected result evidence, monitoring, failure diagnosis, cleanup and limits. Submission requires separate explicit authorization.
- [ ] Run relevant tests, safe dry-run/help, lint and skill validation. Independent script/skill review, fixes/re-review and focused commit.

### Final validation

- [ ] Run a bounded synthetic and cached-real metric smoke using existing assets, without downloads or source training, where feasible; independently audit metrics/artifacts. Do not treat earlier plain pilot as metric validation.
- [ ] Whole-change integration review; fix and independently re-review any findings before final commit.
- [ ] Report exact run command, commits, supported model boundary, CPU/GPU stages, local checks and unverified remote/image/runtime limits. Leave branch/worktree intact; no push/submission.
