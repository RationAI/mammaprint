# Replicate the IBD concept report for MammaPrint

Status: implemented on `feat/concept-explainability`; see the [verification record](../reviews/2026-11-14-concept-explainability.md). The baseline and external-validation notes below record what was known during planning.

## Intent

Explain already trained MammaPrint heads and aggregators using Adam Kukučka's existing IBD concept workflow. Reuse the laboratory implementation and reporting conventions, with the smallest adaptations needed for the MammaPrint data and model interfaces. The user explicitly requested an implementation plan before coding.

The result is a post-hoc dictionary of 16 visual concepts, eight representative tiles from distinct slides per concept, a blinded naming form for a pathologist, concept overlays in xOpat, model-output effects, and reconstruction/prediction fidelity measurements. Fitting the concept dictionary is allowed; retraining the original encoder, aggregator, or head is not part of this work.

## Grounded baseline and provenance

- Proposed implementation base: `RationAI/mammaprint`, current `feat/tiling-values` tip `e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2`. The sweep launcher checks out `feat/tiling-values`. A branch name is mutable: the implementation must recover each selected training run's actual recorded revision and resolved configuration before loading its checkpoint.
- Proposed new branch: `feat/concept-explainability`. Do not reset, stash, or switch the current dirty detached checkout. Establish an isolated checkout at execution time using the app's native worktree mechanism, retaining the pinned base SHA.
- Existing remote `feat/explainability` tip: `2c4bcf65bd6e822013a77fb8b1a591457dceef91`, three commits above the proposed base. It implements custom tile IG, leave-one-out, singleton and patch-flipping methods; do not merge this pipeline. Its data/checkpoint/viewer helpers may be used as references only when they save code.
- Laboratory source: `RationAI/ulcerative-colitis`, `feature/explainability`, pinned at `a6fe08b93219b91a68758a4eaf32159a33f5d4eb`. Preserve source attribution and MIT notice for copied portions.
- The training Sheet's current `Sweep runs!A1:P12` records architecture and code version `2.0`, but does not establish an exact training SHA. Sample completed run `9b74c6abd61a4562a89e512a0a665a7c` is a tissue/level-3/mean/linear/BCE/seed-0 smoke-test candidate, not a scientifically selected winner.
- Checkpoints, resolved run configurations, mounted WSIs, and an actual generated report were not opened in this planning session. No guarantee that every run used a single SHA is implied.

## Approaches and choice

1. Port Adam's flat concept workflow and doctor report; call the existing MammaPrint model for prediction effects. Recommended because it reuses the requested laboratory method while covering nonlinear heads without deriving new attribution formulas.
2. Merge the existing MammaPrint tile-explainer branch. Rejected for this scope: it brings a large different pipeline with custom attribution logic and does not supply the requested concept-naming workflow.
3. Port the full IBD project or the generic concept-bounds research framework. Rejected for this scope: Nancy-specific ensembles, mathematical bounds, institution handling, hierarchy and method sweeps are unnecessary.

## Scientific and behavioral constraints

- Python remains `>=3.12, <3.13`; use the training-compatible Torch/model definitions.
- Use the full signed 2,560-dimensional Virchow2 tile embedding, not only the CLS or patch half. Semi-NMF coefficients are nonnegative; dictionary atoms may be signed. Do not clip input embeddings.
- Default dictionary configuration follows Adam's `semi_nmf_k16`: `n_components=16`, `center=true`, `scale_power=0.0`, `metric.enabled=false`, seed `42`, one epoch, 200 NNLS iterations, online forgetting `0.95`, ridge `1e-6`, unit-row dictionary normalization. Copy his fit/batch/shuffle settings and record them explicitly; overrides must be recorded.
- Fit the dictionary and centering statistics using training embeddings only. Validation is for fidelity assessment/configuration decisions; the test split is for final reporting. Patient/case separation is inherited from the original split and must be checked when the metadata supports it.
- One dictionary per exact embedding-source/level/mask training corpus; reuse it across heads, objectives and seeds using that corpus. Do not transfer the fitted IBD dictionary to breast slides, mix levels implicitly, or refit per checkpoint.
- Preserve existing models, training entrypoints, labels, losses, data cards and split membership. Never append to the source training MLflow run.
- For model effects and fidelity, execute the trained aggregator and head in evaluation/inference mode; keep tile count, tile order and any spatial coordinates unchanged. No closed-form attention or linear-head assumptions in the adapter.
- Classification explains `P(luminal A)` with A=1, B=0; MammaPrint index regression explains the raw scalar in index units. Joint output columns are `[class logit, index]`, not a two-class softmax. Increasing index is not automatically described as increasing risk.
- Use Adam's two-sided concept push, including nonnegative coefficient clipping on subtraction and `step_sds=0.25`; recompute predictions on decoded embeddings. Report add and subtract effects, consistency and reconstruction fidelity. This is the model's response in concept space, not a biological causal claim.
- Select example tiles using Adam's coefficient-share ranking, `n_tiles=8`, `min_total_quantile=0.5`, at most one tile per slide. Log deterministic tie-breaking and any shortage of eligible slides rather than silently duplicating slides.
- Read image tiles using existing WSI libraries at the original embedding geometry; no re-embedding. Derive actual microns from metadata; do not copy the IBD report's approximately 350 micrometre caption.
- Reuse Adam's concept-presence masks with overlap averaging and common per-concept scaling. Presence masks must not be relabelled as signed output-attribution maps.
- Naming form: short name, unclear/mixed, expected output category or no effect, same-as concept and comment; hidden labels/effects until deliberate reveal; freeze expected-category answers after reveal; copy answers without sending them. Blinding is a review convention, not a security boundary.
- The supplied `naming_k16.html` source is not in the inspected branch. Reuse Adam's actual template if supplied. Otherwise port the existing report's layout and add only the small form/reveal/copy behavior described in the Slack message, explicitly documenting that this reproduces behavior, not exact HTML.

## Minimal architecture

Add a separate `ml.concepts` CLI and a small `ml/concept_explainability/` package. Keep upstream numerical functions together with source provenance, separate from the MammaPrint data/model adapter and report rendering. Use existing Hydra, MLflow, Ray, NumPy, pandas, Torch, and WSI-reading/mask-writing libraries. Prefer the existing pinned ratiopath rather than updating the training dependency stack. Add ReportFast only in an optional explainability extra, pinned to the IBD lockfile's revision if required for xOpat sessions.

Process one slide and one concept intervention at a time. Dictionary fitting is streamed; do not accumulate all 2,560-dimensional embeddings or all 32 perturbed bags in RAM. Stream or memory-map the 16-dimensional coefficients. The same forward adapter supports mean, max, gated attention, transformer and spatial transformer if present in the selected checkpoint; do not promise feasibility for a full-resolution transformer without a representative GPU smoke test.

## Outputs and acceptance

Persist dictionary atoms, centering, source/fit manifest, per-tile coefficients and keys, representative tile metadata, reconstruction metrics, original/reconstructed slide outputs, concept effect/presence tables, HTML naming report and concept TIFF masks. Output identifiers must join on dictionary identity, source run, split, slide, tile coordinates and target. Large artifacts remain outside Git and separate from training artifacts.

Accept the port after numerical agreement with the pinned upstream functions on a signed fixture, strict checkpoint loading and prediction parity, training-only fitting, coordinate/order preservation, correct task semantics, distinct-slide galleries, report form/reveal/export checks, one real data pilot with fidelity disclosed, and visual confirmation of tile/overlay alignment when VPN access is available. Poor fidelity must remain visible; do not silently enlarge K or invent a new method to improve it.

## Remaining external checks

1. Recover per-run SHA, resolved Hydra config and checkpoint; compare inference-critical definitions against the base. Stop the affected run on incompatible or unresolvable provenance rather than guessing.
2. Obtain the exact `naming_k16.html` generator and confirm the intended breast expected-category wording with Adam/the reviewing pathologist. Initial proposed wording: luminal A / luminal B / no effect / unsure, with regression effects revealed separately.
3. Confirm mounted paths, actual tile geometry, compatible library APIs and GPU memory using a pilot. These affect execution, not the ability to implement the port.

No implementation branch or worktree has been created as part of planning. No jobs, model training, MLflow writes, or report publication have been performed.
