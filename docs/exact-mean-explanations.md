# Exact tile evidence for mean pooling and a linear head

## Implementation plan

1. Add an opt-in `explain.exact_mean` mode for stored embeddings, `MeanPool`,
   `LinearHead`, identity output activation, and evaluation mode. Reject an
   incompatible model before loading the cohort. Reuse the streamed training
   tile mean already used by the explainer; require its split to be `train`.
2. Compute signed tile contributions analytically, without autograd:
   `c_i = w @ (x_i - training_mean) / N`. Preserve the bias in the reference
   score `z_ref = w @ training_mean + b`. Record the independent full-model
   prediction and the floating-point reconstruction residual. Use index units
   for regression and logit units for BCE; apply sigmoid only to the final logit.
3. Extend lossless tile tables, signed TIFF/GeoJSON layers, viewer labels, and
   default report selection with `exact_mean`. Keep existing methods compatible.
   Do not add redundant patch-flipping calculations for this method: its signed
   ordering equals singleton ordering for these models.
4. Save `mean_evidence.json` with each slide/target's reference, positive and
   negative contribution sums, reconstructed prediction, residual, tile-score
   histogram, and spatial grid summaries. Partition tiles by their origin into
   configurable 8-by-8-stride grid cells; each tile belongs to exactly one cell.
   Store every cell's tile count, bounds, signed sum, and opposing contributions.
   These are descriptive groups, not anatomical regions or removal effects.
5. Include the strongest three tiles on each side of the reference, preserving
   coordinates and exact scores. Read only these small WSI crops at the selected
   pyramid level; embed compact JPEG previews in the evidence JSON. Cache crops
   shared by multiple targets. Missing WSI access must leave numerical evidence
   usable and explicitly mark previews unavailable.
6. Extend the standard RationAI HTML report with a searchable evidence section:
   per-slide expandable cards, reference-to-prediction breakdown, raw-score
   histogram, regional table, representative crops, and existing xOpat links.
   Embed data/previews so report regeneration needs only the small evidence JSON,
   manifest, and slide table, never embeddings or full mask pixels. Existing runs
   without evidence keep their existing report behavior.
7. Make the existing cluster submitter accept explicit CPU, memory, device, and
   `--gpu none`, preserving its current defaults. Prepare two CPU-only test-split
   runs pinned to the pushed commit, plus a one-slide smoke command. Submit no
   jobs during this implementation task.

## Validation and independent review

- Obtain an independent plan review before implementation and record outcomes.
- Test analytic reconstruction against direct model outputs and brute-force
  replacement with the training reference, including bias, nonzero reference,
  negative values, one tile, multiple outputs, and BCE probability semantics.
- Check grid conservation, opposing evidence within a cell, tie handling, empty
  sign selections, missing crops, table/viewer compatibility, old reports, and
  CPU-only job serialization.
- Run a synthetic cohort through the actual runner, TIFF rendering, evidence
  export, and report extension; inspect the final HTML in Chrome.
- Obtain an independent implementation review, address findings, rerun affected
  checks, commit only this work, and push a dedicated branch.

## Scope and interpretation

The frozen embedding extractor is outside this explanation. Training-mean
embeddings are a reference, not healthy tissue. Positive means increasing the
raw model output relative to that reference. Contribution size depends on the
number of included tiles; the singleton histogram retains the unscaled local
score. Spatial sums partition model evidence by tile origin, not by unique WSI
area when tiles overlap. Do not compare numerical magnitudes across regression
and BCE heads. Existing source training/explanation runs are not overwritten.

## Independent plan review

An independent reviewer approved the plan with six refinements, all accepted:
render evidence even without gallery masks; check absolute plus relative residual
tolerance; explain scale and tile-count dependence; bound and reuse JPEG crops;
separate numerical, mask, and crop failures; verify CPU-only manifests using the
installed cluster library. Reference replacement is an algebraic model
counterfactual, not a biological intervention.

The two requested source runs are recovery runs. Port the existing, previously
used opt-in resolved-config loader from the November 17 submission package,
including checkpoint/config hash assertions, and pass those options through the
submitter. This preserves their exact saved data/model settings without needing
historical Hydra task overrides. Add checks for this concrete submission path.

## Cluster submission plan

Use `python -m scripts.ml.submit_exact_mean_reports` from this checkout, with the
environment installed by `uv sync --frozen`. It previews by default; `--submit`
creates one job. Always use the full pushed commit SHA for `--revision`.

1. Preview a one-slide smoke run for each model using `--model regression` or
   `--model bce`, `--revision <pushed-sha>`, and `--slide-id <test-slide-id>`.
   Add `--submit` only when ready. Use the same test slide for both models.
2. Check the actual cluster WSI crop coordinates, loaded images, TIFF alignment,
   viewer layers, evidence reference, and reconstruction residual. The local
   synthetic test mocks native WSI IO because OpenSlide is unavailable locally.
3. Preview and then submit the full test split for each model by omitting
   `--slide-id`. These two jobs can run in parallel. Each makes a new MLflow run
   and writes the extended standard report to `explanations/report.html`.
4. Confirm 308 successful numerical slides and 308 overlays per model, no skipped
   training-reference slides, and all reconstruction residuals within tolerance.
   Recompute metrics from `slide_keys.csv` to verify regression F1 0.7922438 /
   MAE 0.1786979, and BCE F1 0.7537994 / accuracy 0.7370130 (tolerance 1e-5).
   If those differ, investigate rather than treating the new explanations as
   directly comparable. Both class boundaries remain raw output >= 0.

Per full job: 4 requested/limited CPUs, 8 GiB RAM, no GPU, explicit CPU inference,
and the existing secure.DATA / secure.PROJECTS mounts. The cluster library's
default shared-memory volume allows up to 4 GiB, charged within the 8 GiB RAM
limit when used (not an additional RAM reservation). Temporary
embeddings and generated artifacts go beneath the secured shared scratch root
`/mnt/projects/mammaprint/exact-mean-explanations`; allow 80 GiB free per job
(160 GiB with both running). This is a planning budget, not a PVC reservation or
hard quota. No explicit node ephemeral-disk request is made by this submitter;
leave about 15–20 GiB per job for checkout, environment, and package cache.

The earlier regression run's saved slide table measured 308 slides / 631,812
tiles, about 108 seconds total attribution and 49 minutes rendering two methods.
The new third mask pair and additional crops suggest roughly 1–2 hours of
processing per full job, plus training-baseline downloads, environment setup,
uploads, and queue time. Budget 2–3 hours after scheduling, then refine from the
smoke run. This is an extrapolation, not a benchmark of the new full workflow.
Prior CPU workers used roughly 2–3 GiB RSS; 8 GiB is a request with headroom,
not a new measured peak. Concurrent allocation is 8 CPUs, 16 GiB RAM, zero GPUs.

Integrated Gradients and attention stay disabled. Single-tile output, tile
removal, and the new exact contribution heatmaps stay enabled. Patch flipping
is disabled because these mean + linear rankings are equivalent and the older
runs already contain their faithfulness curves. Neither source training nor
previous explanation runs are changed. No cluster submission is part of the
implementation/commit/push task.

## Implementation review and validation

The independent implementation reviewer approved the final changes after
independently rerunning all 10 focused Python tests. Two findings were fixed:
`--methods single` now omits the exact-evidence section, and preview IO has a
separate runtime field included in total slide runtime.

Validation covers direct prediction reconstruction, brute-force reference
replacement, agreement with IG, bias/nonzero-reference/single-tile/multi-output
cases, BCE probability handling, region conservation/opposing contributions,
deterministic ties, crop-coordinate forwarding/reuse/failure isolation, unsafe
HTML/URL escaping, real TIFF and GeoJSON output, standard RationAI template
integration, numerical-only fallback, report method filtering, saved-graph
composition, and CPU manifest construction through the installed job library.
An additional 13 existing attribution/patch-flipping checks passed against this
checkout. Ruff and whitespace checks passed.

Read-only live dry runs also downloaded both requested MLflow checkpoints and
saved configurations, verified their pinned SHA-256 hashes, and loaded the exact
model state strictly on CPU. Both passed with MeanPool / LinearHead / one output;
the recovered regression and BCE checkpoints have epochs 48 and 41 respectively.
These checks created no MLflow runs and performed no cohort inference.

The rendered synthetic report was inspected in Chrome. Its search, embedded
images, histogram, regional expansion, and crop links passed automated browser
checks. The preview uses simulated images, not patient tissue. Native WSI IO,
real viewer access/alignment, and the new full-cohort peak RAM remain unverified
until the cluster smoke tests. No cluster jobs were submitted during this work.
