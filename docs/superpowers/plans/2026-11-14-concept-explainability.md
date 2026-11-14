# MammaPrint IBD Concept Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan does not authorize model retraining or blanket cluster execution.

**Goal:** Replicate Adam Kukučka's 16-concept IBD explainability and doctor-naming workflow for the existing MammaPrint checkpoints, with minimal new integration code.

**Architecture:** Fit Adam's existing centered semi-NMF dictionary on the training Virchow2 embeddings, reuse his concept encoding, tile selection and reporting conventions, and evaluate concept interventions through the original trained MammaPrint model. Add a standalone CLI; keep numerical code copied from the pinned laboratory source separate from data/model adapters. Use a shared dictionary for runs with the exact same training corpus.

**Tech Stack:** Python `>=3.12, <3.13`, Torch, Hydra/OmegaConf, MLflow, Ray Data, NumPy, pandas/PyArrow, scikit-learn, existing WSI libraries; optional ReportFast for xOpat links.

**Spec:** [`../specs/2026-11-14-concept-explainability-design.md`](../specs/2026-11-14-concept-explainability-design.md). Read the spec first; it contains the user's scope, source evidence and deferred external checks.

## Global Constraints

- Base the new `feat/concept-explainability` branch on `e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2`, the current `feat/tiling-values` tip, subject to per-run provenance verification. Do not equate the branch tip with every run's training SHA.
- Do not alter or carry over the current uncommitted sweep edits, existing worktrees or detached HEAD. Create isolation at implementation time; no branch was created during planning.
- Copy numerical methods from IBD commit `a6fe08b93219b91a68758a4eaf32159a33f5d4eb`; preserve attribution/license and list every adaptation. Do not implement a new XAI algorithm or merge the large `feat/explainability` attribution pipeline.
- Keep original encoders, heads, aggregators, losses, metrics, training entrypoints, dataset cards and split membership unchanged. Load model weights strictly; never use `strict=False` to hide incompatibility.
- Use full signed 2,560-dimensional Virchow2 embeddings; `n_components=16`, `center=true`, `scale_power=0.0`, `metric.enabled=false`, seed `42`, one epoch, 200 NNLS iterations, online forgetting `0.95`, ridge `1e-6`, unit-row dictionary normalization.
- Fit statistics and dictionaries on training data only. Reuse a dictionary only when embedding source, level, mask variant and usable training slide roster match. The IBD dictionary itself is not reused.
- Keep tile count/order and spatial coordinates fixed during reconstruction/intervention; run the original model in eval/inference mode. Process one slide and one intervention at a time.
- Classification target is `P(luminal A)`, A=1/B=0; regression target is the raw MammaPrint index. Joint outputs are `[class logit, index]`; no two-output softmax or unverified high/low-risk wording.
- Report source-model fidelity and two-sided intervention consistency. Do not hide poor fidelity or silently change K/method.
- Source training MLflow runs are read-only. New artifacts go in a new explanation run or explicitly selected local output directory, outside Git. Publication and cluster execution are separate operational steps.

## Review Focus

1. A checkpoint with a different model definition or unresolved source configuration must fail before reporting, not receive an approximate load. Covered in Task 1.
2. Joint/regression runs may have different usable slide rosters even with identical embedding URIs; dictionary reuse must compare corpus identities. Covered in Task 2.
3. LayerNorm, MLPs, transformers and spatial coordinates invalidate IBD's analytic prediction shortcuts; preserve the model's native forward computation. Covered in Task 3.
4. Tiles/overlays from a different level, row order, slide or physical footprint can look plausible while being wrong. Covered in Tasks 2 and 4.
5. A blinded report can accidentally reveal labels, lose answers on reveal or export the wrong target wording; test the state transition and response payload. Covered in Task 4.

## What is established and what remains

The sweep launcher currently uses `GIT_REF = "feat/tiling-values"`. GitHub confirms its tip is `e86923d...`; the latest existing tile-explainer branch is three commits above that base. The model computation files should remain byte-identical to the training base throughout this feature.

The current training Sheet has renamed the old experiment tab to `old - experiments`; use the current `Sweep runs` tab, not the stale skill tab name. `Sweep runs!A1:P12` contains completed runs and `code_version=2.0`, not exact Git SHAs. Row 4 identifies `9b74c6abd61a4562a89e512a0a665a7c` as tissue/level-3/mean/linear/BCE/seed-0: use it only as a deterministic operational smoke-test candidate. A pilot must not be chosen as the best performer on the test split.

We have the method, numerical code, default configuration, example selection, effect definition, fidelity checks and report structure. We have not verified actual checkpoint/config contents or the published HTML. The specific annotation form `naming_k16.html` is not present in the inspected IBD tree; ask Adam for its generator if exact presentation parity matters. Otherwise its behavior is fully described by the supplied Slack message, and the existing report provides the base layout.

## Source reuse map

All paths in this table are in [RationAI/ulcerative-colitis at the pinned commit](https://github.com/RationAI/ulcerative-colitis/tree/a6fe08b93219b91a68758a4eaf32159a33f5d4eb/explainability). Training provenance was checked against [the training ledger](https://docs.google.com/spreadsheets/d/1k2fB0s5-l6TUmzkzi0gqRyqDndiEbbT9JGVzuVytQ_w/edit#gid=1779572960), bounded ranges `Sweep runs!A1:P12` and `N3:P8`.

| Source | Reuse | Necessary adaptation |
|---|---|---|
| `explainability/factorization.py` | `nnls_batch`, `kmeans_init`, `reseed_dead_components` | Imports/module path only; preserve numerical bodies. |
| `explainability/nmf_fit.py` | `iter_embedding_batches`, `fit_semi_nmf_online`, `gauge_fix_dictionary` | Feed MammaPrint parquet dataset/config; remove Nancy checkpoint/output-metric wiring. |
| `explainability/concepts.py` | `ConceptAutoencoder.encode/decode` | Local dictionary paths and saved manifest; no IBD institution loader. |
| `explainability/tree_view.py` | `code_share`, JPEG/tile-link/CSS helpers | Flat report only; omit the tree. |
| `explainability/concept_gallery.py` | `select_tiles`, image-reading convention | MammaPrint slide identifiers and metadata; retain share ranking and one tile per slide. |
| `explainability/pathologist_view.py` | Two-sided push definition and report conventions | Replace Nancy ensemble evaluation with native model forward calls; task-specific labels/units. |
| `explainability/concept_masks.py`, `tiles.py` | Presence mask averaging/scaling and xOpat sessions | Geometry/schema and artifact locations; use existing compatible library APIs. |

Do not copy IBD `ModelWeights`, analytic tile-classifier/attention reconstruction, Nancy grading, ensembling, tree growth, Lipschitz bounds, OLS surrogate heads, method sweeps, or its whole preprocessing pipeline.

## Proposed file map

| File | Responsibility |
|---|---|
| `ml/concepts.py` (new) | Hydra CLI dispatching `action=fit` or `action=explain`; new-run logging. |
| `ml/concept_explainability/__init__.py` (new) | Package marker only. |
| `ml/concept_explainability/upstream.py` (new) | Copied numerical/selection functions above, with pinned provenance. |
| `ml/concept_explainability/data.py` (new) | Resolve exact source dataset, identify corpus, stream embeddings, retain tile keys/geometry and read representative RGB tiles. |
| `ml/concept_explainability/model.py` (new) | Read training run/config/checkpoint and call the original model with correct output semantics. |
| `ml/concept_explainability/evaluation.py` (new) | Dictionary orchestration, streamed encoding, fidelity and existing two-sided concept intervention. |
| `ml/concept_explainability/report.py` (new) | Adapted flat naming report, effect tables, masks and xOpat links. |
| `configs/concepts.yaml` (new) | One small configuration for actions, source run, dictionary/output directories and copied IBD defaults. |
| `tests/test_concept_sources.py`, `tests/test_concept_evaluation.py`, `tests/test_concept_report.py` (new) | Standard-library unittest integration/adaptation tests, including upstream parity fixtures; follow the existing test style without adding pytest. |
| `docs/concept-explainability.md` (new) | Reproduction commands, scientific interpretation, artifacts and differences from IBD. |
| `third_party/ibd-explainability-NOTICE.txt` (new) | Original MIT notice, source URLs/commit and copied symbols. |
| `pyproject.toml`, `uv.lock`, `README.md` (modify only as needed) | Optional explainability dependencies and short usage link. |

Avoid a generic plugin framework, new training abstractions, another dataset class, a new model architecture, or a custom dashboard. A separate cluster submitter is not needed for the first implementation; reuse the established `submit_job` pattern manually with the pinned feature SHA after the pilot command is ready.

## Task 1: Pin the base and prove checkpoint reproduction

**Files:** `model.py`, `tests/test_concept_sources.py`; no model/training-file edits.

**Interfaces:** `SourceRun(run_id: str, training_sha: str | None, config: DictConfig, checkpoint_path: Path | None, checkpoint_sha256: str | None)`; `load_source_run(run_id: str, cache_dir: Path, *, download_weights: bool) -> SourceRun`; `load_frozen_model(source: SourceRun) -> MammaprintModule`; `model_outputs(module: MammaprintModule, bag: Tensor, label_mode: str) -> Tensor` returns raw logit/index column order.

- [ ] Establish a clean isolated checkout using the app's native worktree tool with `ref=e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2`; create the proposed branch without modifying the original workspace. Verify the fetched ref still resolves to that SHA. Do not base this work on the unrelated multiscale worktree or merge `feat/explainability`.
- [ ] Add tests named `test_source_run_load_is_read_only`, `test_checkpoint_requires_exact_graph`, `test_missing_resolved_config_fails`, `test_joint_outputs_are_logit_and_index`. Assert no source-run logging calls occur, missing/unexpected state keys fail, absent resolved config fails clearly, and joint outputs remain two independently interpreted columns. Run the new tests once to establish the failure, then implement the small adapter.
- [ ] Read `configs/config-resolved.yaml`, MLflow run metadata/tags and `checkpoints/best/checkpoint.ckpt` using the existing artifact conventions in `ml/resume_test.py`. Reuse read-only snippets rather than its resume entrypoint, which writes to the training run. Require the resolved config for v1; do not add a broad recovery engine for missing historical artifacts.
- [ ] Inspect `mlflow.source.git.commit` or other recorded revision; resolve its exact SHA and compare inference-critical model definitions/configuration with the implementation base. If unknown, recover from the training run's logged source/config/logs; mark unresolved provenance explicitly and stop final scientific reporting for that run. Different SHAs are acceptable only after establishing identical inference semantics and strict load/parity.
- [ ] Instantiate `source.config.ml` with the existing Hydra pattern, load the state dictionary strictly, set `.eval()` and freeze parameters. Compute outputs through the native encoder/aggregator/head; do not reconstruct weights into an IBD-specific NumPy model.
- [ ] Verify original versus adapter raw outputs on the same bag with `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` in FP32, documenting any device-dependent need for a justified tolerance. Do not use equality of aggregate metrics as a substitute for prediction parity.
- [ ] Run `uv run python -m unittest discover -s tests -p test_concept_sources.py -v`. Expected: all tests pass. Record the representative real checkpoint's SHA, checksum, config and parity result when mounted data are available.
- [ ] Commit only this independently reviewable adapter and its tests.

## Task 2: Port the dictionary and data adapters

**Files:** `upstream.py`, `data.py`, `configs/concepts.yaml`, notice, source/evaluation tests, dependency metadata only as necessary.

**Interfaces:** `corpus_identity(source: SourceRun) -> dict[str, Any]`; `embedding_dataset(source: SourceRun, split: str) -> ray.data.Dataset`; `iter_slides(source: SourceRun, split: str) -> Iterator[SlideInput]`, where `SlideInput` has `slide_id`, `(N,2560)` embeddings, separate `(N,2)` coordinates, label and geometry; `fit_dictionary(source: SourceRun, config: DictConfig, output_dir: Path) -> Path`; `load_dictionary(directory: Path) -> ConceptAutoencoder`.

- [ ] Add a small signed-embedding fixture, two training slides and one held-out slide. Add tests `test_ibd_nnls_and_decode_parity`, `test_fit_uses_train_only`, `test_corpus_identity_includes_usable_slide_roster`, `test_encode_preserves_tile_keys`, `test_empty_or_wrong_width_corpus_fails`. Include a held-out extreme value and assert it cannot affect centering; reject zero-row/nonfinite/wrong-width inputs.
- [ ] Copy the specified upstream numerical functions and source notice. Compare outputs against expected values generated by the pinned upstream code on the same fixture, not against a second hand-written implementation. Maintain NumPy dtypes, centering order, signed atoms, 200 NNLS iterations, reseeding and final normalization.
- [ ] Instantiate the original `SingleScaleDataset`/`SpatialScaleDataset` for artifact and usable-slide resolution, but read the original parquet rows retaining `x`, `y`, original row index and slide stem. Existing training metadata carries only `slide_id`; the explanation adapter must recover coordinates from parquet, without changing the training dataset.
- [ ] Feed the eligible parquet paths to Ray for Adam's existing batching/shuffling. For global fit statistics, use the exact streaming sum/count mean already used in `token_statistics.py`; because scale power is zero, the scale vector is exactly ones. Omit percentile/t-digest calculation and its dependency. Record this omission as removal of unused preprocessing, not a changed factorization.
- [ ] Copy batch size `25600`, local shuffle buffer `250000`, seed `42`, epochs `1`, NNLS iterations `200`, forget `0.95`, ridge `1e-6`, centering and final unit-row normalization. Treat these as explicit defaults; expose bounded resource overrides without changing scientific defaults invisibly. Start the full fit with Adam's conservative 8-CPU Ray limit. Verify the buffer's real memory footprint before running it on larger cohorts.
- [ ] Persist `h.parquet`, `shift.npy`, manifest and per-slide coefficient parquet files keyed by slide/row/x/y. Preserve the upstream reconstruction `z_hat = shift + v @ H`; large coefficient arrays are streamed or memory-mapped, not held with the complete embedding corpus. Save corpus identity including exact split artifact URIs and sorted usable training roster.
- [ ] Declare directly imported numerical packages explicitly in an optional extra while retaining currently locked compatible versions. Current MammaPrint locks ratiopath `1.3.1`; IBD uses `1.4.2`. Verify the required tile/mask APIs first and prefer MammaPrint's existing readers rather than upgrading the training stack. Do not add a dependency on the entire IBD application.
- [ ] Run the relevant source/evaluation tests and a small local dictionary fit. Expected: deterministic fixture parity, no test/validation leakage, no row-key misalignment, and signed values retained. Commit the dictionary/data port and notice.

## Task 3: Evaluate fidelity and concept effects through each frozen model

**Files:** `model.py`, `evaluation.py`, `tests/test_concept_evaluation.py`.

**Interfaces:** `encode_slide(slide: SlideInput, dictionary: ConceptAutoencoder) -> np.ndarray`; `evaluate_concepts(source: SourceRun, dictionary_dir: Path, split: str, output_dir: Path) -> Path` writes slide outputs, fidelity and concept effect/presence tables.

- [ ] Add adaptation tests covering mean/max/gated-attention/transformer with linear and MLP heads, all three label modes, and spatial-transformer coordinate preservation. Synthetic bags are small; no GPU is required for these tests. Add `test_zero_push_matches_reconstructed_forward`, `test_joint_probability_uses_sigmoid_only_on_column_zero`, `test_spatial_coordinates_remain_unchanged`, `test_zero_spread_concept_has_zero_effect`, `test_subtraction_stops_at_zero`, and `test_same_dictionary_is_reused_across_matching_runs`.
- [ ] For every slide, evaluate original embeddings `Z` and decoded embeddings `Z_hat` through the same original model. Compute embedding reconstruction RMSE and output fidelity separately: raw-logit RMSE, luminal prediction agreement/probability differences for classification, raw-index RMSE for regression. Use existing metric helpers where applicable; disclose undefined correlations and missing groups rather than inventing values.
- [ ] Compute concept shares `v_ik / max(sum_l v_il, 1e-12)` using Adam's helper. For each concept, compute the SD (`ddof=1`) across slides of its mean share on the selected explanation cohort, exactly as his report does. This cohort calibration is descriptive, not fit-time dictionary information; record its split. Require at least two slides for SD-based effects, otherwise emit an explicit insufficient-cohort result.
- [ ] Port Adam's exact push size `d_i = 0.25 * sd_k * sum_l v_il`. Added code is `v_ik + d_i`; subtracted code is `max(v_ik - d_i, 0)`. Decode each modified bag, preserve spatial coordinates and call the original model again. Baseline effects are against `f(Z_hat)`, not `f(Z)`; keep `f(Z)` beside them for fidelity. Process one intervention at a time.
- [ ] Apply sigmoid to the classification logit for displayed `P(A)` effects, in percentage points. For B-group cells use `P(B)=1-P(A)`. For regression keep effects in raw index units, summarized over the cohort; for joint models emit both targets separately. Preserve both add/subtract arms, their average, direction disagreement flags, consistency fraction and clipping fraction. Do not carry the IBD grade thresholds or minor-effect cutoff across incompatible units.
- [ ] Use predicted A/B groups for classification presence/effect summaries, matching the model-grouped IBD convention; identify ground-truth labels separately. For regression use overall concept presence/effect and retain per-slide values. Do not create arbitrary low/high-index bins.
- [ ] Record unsupported resource cases as explicit failures; do not subsample trained bags or silently switch to a cheaper aggregator. A transformer requires roughly 34 forwards per slide for original, reconstructed and 32 interventions, with quadratic attention cost; pilot before a full cohort.
- [ ] Run `uv run python -m unittest discover -s tests -p 'test_concept_*.py' -v`. Expected: parity and intervention tests pass for the supported architecture matrix. Commit this adapter; no new attribution engine is introduced.

## Task 4: Reproduce the doctor-facing naming report and overlays

**Files:** `report.py`, `tests/test_concept_report.py`, small report settings in `configs/concepts.yaml`.

**Interfaces:** `build_report(source: SourceRun, dictionary_dir: Path, evaluation_dir: Path, output_dir: Path) -> Path` returns `naming_k16.html`; outputs join on `dictionary_id`, `source_run_id`, `split`, `slide_id`, row index, coordinates and concept ID.

- [ ] Request Adam's generator for the exact linked `naming_k16.html`. If available, pin and port that template. Otherwise adapt the checked-in `pathologist_view.py` layout plus the Slack-described form. Keep a note of which route was used; this dependency does not prevent the numerical port.
- [ ] Add tests `test_eight_examples_use_distinct_slides`, `test_insufficient_eligible_slides_is_reported`, `test_blind_report_has_no_visible_labels_or_effects`, `test_reveal_preserves_answers_and_locks_expectation`, `test_answer_export_preserves_dictionary_identity`, `test_mask_and_tile_geometry_match_embedding_level`. Check duplicate/self `same_as` values and HTML escaping for paths/names/comments.
- [ ] Reuse Adam's selection: rank by coefficient share, threshold total code at quantile `0.5`, take up to eight distinct slides per concept. Preserve deterministic tie handling and report shortages. Filter to eligible tiles before selecting the top rows; if fewer than eight eligible slides exist, show fewer examples and a shortage notice. This is an explicit correction to the source helper's zero-score fallback, not a different ranking method.
- [ ] Read the exact embedding tile footprint from original WSIs using existing WSI readers. Render unnormalized RGB thumbnails. Caption the measured physical width at that level; MammaPrint's level-3 card says 224 px at 1.94 micrometres/px (about 435 micrometres nominally), so the IBD approximately 350 micrometre description cannot be copied. Verify actual slide metadata before making physical-size claims.
- [ ] Form fields: concept name, unclear/mixed, expected luminal A/B/no effect/unsure, same-as concept, comment. This A/B wording is a proposed breast adaptation pending pathologist confirmation. Regression effects remain separate and are revealed with other model information. Include dictionary/source-run identity in copied answers.
- [ ] Keep labels/effect numbers hidden initially; explicit reveal freezes the expected-category responses while preserving other entered text. Use the existing report's static HTML approach and a small JS copy/reveal handler; no server or storage system. Warn that reload loses edits and provide copyable text if clipboard access is unavailable. Do not send responses to Slack or any other service.
- [ ] Reuse presence-mask overlap averaging and shared per-concept quantile scaling (`0.5`, `0.999`). Reserve no-tile pixels separately, retain physical alignment and expose only the selected concept by default. Use the actual metadata downsample; do not carry the IBD value `9` into a scanner with a different pyramid.
- [ ] For clickable slides/overlays, reuse the existing xOpat session and viewport pattern. If needed, add optional `report-fast` pinned to IBD's lock revision `810c09fed2fc968ffe715d81b56b3b0d2052ec47`. Validate compatibility with the current WSI/mask libraries first; a small schema adapter is preferable to a dependency-stack migration.
- [ ] Run report tests, render a local HTML fixture and inspect the form state transitions, exported answers and tile placement. Test a TIFF readback for overlap/geometry. When VPN access resumes, open one real example and verify tile viewport plus concept overlay alignment. Commit the report port; do not claim live viewer QA before performing it.

## Task 5: Add the small CLI and run a bounded pilot

**Files:** `ml/concepts.py`, `configs/concepts.yaml`, documentation and README; existing tests only as needed for orchestration coverage.

**Interfaces:** Hydra options `action=fit|explain`, `source_run_id`, `dictionary_dir`, `split=train|val|test`, `output_dir`, `device=cpu|cuda|auto`; `source_run_id` is required. `fit` requires training split; `explain` requires a fitted dictionary. `explain` performs evaluation then report generation; no separate general workflow engine.

- [ ] Wire the proven functions into the existing Hydra/MLflow pattern. Always create a new explanation run with source training run/SHA/checksum and dictionary provenance; never replay `logger.run_id` or source-run identity. A local-output mode must remain possible for synthetic tests.
- [ ] Record corpus identity, upstream numerical source SHA, implementation SHA, copied settings, dependency versions, fit/evaluation splits, patient/case audit outcome, geometry and any behavior differences in a manifest. Store source config/checkpoint checksums without credentials. Hash dictionary atoms/centering to establish dictionary identity.
- [ ] Use one dictionary fit for a corpus, then explain multiple explicitly selected checkpoints with it. A short shell loop over selected run IDs is sufficient; no new sweep scheduler or automatic best-run selection. Start with one seed/run per desired architecture and expand only after the pilot's runtime and fidelity are understood.
- [ ] Add a synthetic end-to-end test that runs `fit` then `explain`, opens persisted tables/report and proves the input state dictionaries and source-run records were not mutated.
- [ ] Document these proposed commands; they are new interfaces to implement, not commands available on the current checkout:

```bash
uv run --extra explainability -m ml.concepts action=fit \
  source_run_id=9b74c6abd61a4562a89e512a0a665a7c split=train \
  output_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/dictionary

uv run --extra explainability -m ml.concepts action=explain \
  source_run_id=9b74c6abd61a4562a89e512a0a665a7c split=val \
  dictionary_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/dictionary \
  output_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/validation/9b74c6abd61a4562a89e512a0a665a7c
```

- [ ] Complete source provenance/strict-load checks before running that real pilot. Use the mounted cluster environment for large artifacts and WSIs. Log resource limits and pilot elapsed time/peak memory; no runtime guarantee follows from the small synthetic tests.
- [ ] Keep pilot subset results clearly marked. A full-fit dictionary must use the exact training corpus; do not present a sample-fit smoke dictionary as the final result. Run validation fidelity and eight-slide gallery before any final test report. Test evaluation is a final reporting step, with the dictionary/settings frozen.
- [ ] Run `uv run python -m unittest discover -s tests -p 'test_concept_*.py' -v`, targeted lint on new files, and the real pilot when data access permits. Inspect `git diff <base> -- ml/models ml/train.py ml/data configs/data configs/experiment/ml`; expected output is empty. Review dependency changes to ensure model stack versions have not drifted.
- [ ] Commit the CLI/docs after verification. Report separately: source parity, checkpoint parity, offline report QA, real pilot fidelity, runtime/memory and live xOpat QA. Any outstanding access-dependent checks must remain explicit.

## Artifact layout

```text
<corpus>-k16/
  dictionary/
    h.parquet
    shift.npy
    manifest.json
    coefficients/train/<slide>.parquet
  validation/<source_run_id>/
    coefficients/<slide>.parquet
    slide_outputs.parquet
    fidelity.parquet
    concept_effects.parquet
    concept_presence.parquet
    representative_tiles.parquet
    naming_k16.html
    masks/concept_<id>/<slide>.tiff
    manifest.json
```

Cached encodings/gallery selections can be reused across runs when dictionary, split, tile-key roster and geometry match. Do not build a new distributed cache subsystem; use the saved artifacts and identity checks.

## Completion criteria and limits

- The selected checkpoint reproduces its original inference graph/outputs; every supported architecture uses that graph for intervention.
- The dictionary matches Adam's pinned numerical method and defaults, with training-only fit and corpus-specific identity.
- The report has 16 concepts and up to eight eligible distinct-slide examples per concept, with shortages visible, blinded answers/export and physically aligned concept overlays.
- Model effects are in the correct units and shown beside measured fidelity; anatomical names come from the pathologist rather than automatically inferred labels.
- Original training code/artifacts remain unchanged; the feature is independently runnable on a new branch.
- A real cohort report is not considered validated until checkpoint/data access, numerical fidelity, resource behavior and viewer alignment have been checked. If K=16 has poor fidelity for a model, report that limitation; choosing another K or method is a separate scientific decision.

## Optional message to Adam

> Ahoj Adam, chceme v MammaPrint zopakovať váš IBD postup so 16 semi-NMF konceptmi a reportom pre Fabiána. Numerický kód sme našli vo vetve `feature/explainability`. Vedel by si poslať generátor konkrétneho `naming_k16.html` a potvrdiť, z ktorého commitu/konfigurácie vznikol? Naša plánovaná adaptácia ponechá fit konceptov a obojstranné zmeny koeficientov, ale predikcie prepočíta cez naše pôvodné agregátory a lineárne/MLP hlavy. Máte už takýto adaptér, ktorý by sme mohli priamo prevziať? Na slepé pomenovanie zatiaľ navrhujeme očakávaný luminal A/B, bez vplyvu alebo neisté; samotný MammaPrint index by sme ukázali osobitne až po odhalení výsledkov.

## Planning verification

The design received an independent read-only builder check: no consequential unanswered planning questions remained. The plan was checked against the minimal-change requirement, the source/model mismatch, split leakage, target semantics, geometry, report behavior and deferred access checks. No application code, dependencies, branch/worktree, jobs or remote artifacts were changed during planning.
