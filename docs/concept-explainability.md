# MammaPrint concept explainability

This is a post-hoc port of Adam Kukučka's flat IBD concept workflow, pinned to
[`ulcerative-colitis` commit `a6fe08b9`](https://github.com/RationAI/ulcerative-colitis/tree/a6fe08b93219b91a68758a4eaf32159a33f5d4eb/explainability).
The numerical semi-NMF/NNLS routines are copied with their MIT notice under
`third_party/`. They extract 16 visual patterns from the existing signed
2,560-dimensional Virchow2 embeddings. Original models are loaded strictly and
remain frozen.

The integration calls the trained MammaPrint aggregator and head when concept
coefficients are changed. It does not use Nancy-specific ensemble formulas or
assume a linear head. Mean, max, gated attention, transformer and spatial
transformer aggregators are covered by the local model tests.

## Setup and provenance

```bash
uv sync --frozen --extra explainability
```

Use the original cluster image/storage mounts for real slides. WSI reading and
mask writing require working OpenSlide and libvips installations. Native library
installation is environment setup; the workflow does not upgrade them or the
training model stack. The extra adds the pinned ReportFast library and declares
the numerical packages it imports; existing package versions are preserved.

Each source run must provide `configs/config-resolved.yaml`,
`checkpoints/best/checkpoint.ckpt` (for explaining), and a recorded Git revision.
The normal `mlflow.source.git.commit` tag supplies that revision. If an old run
does not have the tag, recover the actual SHA from its logs/source artifacts and
pass `training_sha=<40-character SHA>`. It cannot override a conflicting tag.
Fetch that historical Git object before running the tool if it is not local.
The tool requires identical inference/data code and strict state-dictionary
loading, and accepts the Identity output activation used by the supported
training configurations; other activations fail explicitly. Do not supply the
latest branch SHA as a guess.

All source run operations are read-only. Results are logged in a new experiment
named `MammaPrint concepts`, or written locally with `log_to_mlflow=false`.
The run's original logger identity is never replayed. Output directories must be
empty and dictionaries are immutable. Keep output data outside Git.
Unset an inherited `MLFLOW_RUN_ID` before logging: the CLI rejects it because
MLflow could otherwise resume and alter that run.

## Fit a dictionary once

Replace the example run ID with an explicitly selected run. The example is a
mean/linear/classification seed-0 operational pilot, not a test-selected winner.

```bash
uv run --frozen --extra explainability -m ml.concepts \
  action=fit source_run_id=9b74c6abd61a4562a89e512a0a665a7c split=train \
  output_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/dictionary
```

Defaults match Adam's centered, unscaled, non-metric semi-NMF fit: 16 concepts,
seed 42, one epoch, 25,600-tile batches, a 250,000-row local shuffle buffer,
200 NNLS iterations, forgetting 0.95, ridge 1e-6 and unit-row atoms. Statistics
and fitting use training data only. No clipping is applied to signed embeddings.
An exact running mean replaces the IBD percentile-statistics job because its IQR
scale would be raised to zero; no t-digest or percentile preprocessing is needed.

The Ray job uses 8 CPUs by default. The shuffle buffer can consume several GB;
measure the pilot before choosing cluster memory. Resource overrides, e.g.
`fit.nmf.batch_size=4096 fit.nmf.shuffle_buffer_size=32768`, are saved in the
dictionary manifest. They change batch/shuffle behavior and are not silently
treated as exact reproduction of the original execution settings.

Reuse this dictionary across heads/objectives/seeds only when the exact input
training corpus matches: same artifact URI, level, mask variant, geometry and
usable slide roster. Label filtering can change that roster between objectives.
Fit a separate dictionary for a different corpus; never use the fitted IBD
dictionary on the breast cohort.

## Explain and create the naming report

Start with validation, leaving the dictionary/settings fixed for final test
reporting:

```bash
uv run --frozen --extra explainability -m ml.concepts \
  action=explain source_run_id=9b74c6abd61a4562a89e512a0a665a7c split=val \
  dictionary_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/dictionary \
  output_dir=/mnt/projects/mammaprint/explainability/concepts/tissue-l3-k16/val/9b74c6abd61a4562a89e512a0a665a7c
```

Set `device=cpu` for an appropriately sized local test, or `device=cuda` in a GPU
job. Native transformer inference has quadratic attention cost and concept
intervention needs about 34 forward passes per slide. Tiles are never silently
subsampled; pilot one suitable model/cohort before a sweep. A normal shell loop
over selected source run IDs is sufficient; there is no new scheduler.

The source mapping CSV and per-split embedding files must still be the same
versions used by training. The tool checks current slide and available patient/
case disjointness; a recorded run config containing a mutable file path alone
cannot prove historical label contents. Retain the original label table for the
cluster validation.

Outputs include:

- `h.parquet`, `shift.npy`, immutable dictionary ID and fit manifest;
- per-slide concept coefficients with original row indices and `(x,y)` keys;
- `slide_outputs.parquet` and `fidelity.parquet`, comparing original and
  reconstructed predictions;
- per-slide and grouped concept effect/presence tables;
- `representative_tiles.parquet`, `naming_k16.html`, and presence masks;
- manifests with source/dictionary identity, checksums, settings, dependency
  versions, split-audit outcome and actual slide geometry.

Only representative slides receive masks, with all 16 layers available. xOpat
links use the shared ReportFast session code; a published explanation run uses
MLflow artifact IDs. With logging disabled, links point at local/mounted output
paths; the remote viewer can only open paths available to its server.

## What the numbers mean

For each slide, the dictionary reconstructs `Z_hat = mean + coefficients @ atoms`.
Fidelity compares `f(Z)` against `f(Z_hat)`; embedding RMSE is
`sqrt(sum of squared feature errors / number of tiles)`, as in the IBD script.
Poor fidelity remains visible and must limit interpretation. A visually coherent
gallery alone does not show that the concepts explain the original model well.

For each concept, the code increases and decreases its coefficient on every tile
by Adam's `0.25 * SD(slide mean concept share) * total tile code`, clipping the
decrease at zero. Both perturbed embeddings go through the whole frozen model.
Effects compare with `f(Z_hat)`, not directly with `f(Z)`. They are descriptive
model responses in reconstructed concept space, not biological causal effects.
The SD is calibrated on the explicitly named explanation cohort; this is not
dictionary fitting. A cohort with fewer than two slides is marked insufficient.

- Classification: A=1/B=0. Displayed effects are percentage points of P(A) in
  predicted-A slides and P(B) in predicted-B slides.
- Regression: effects are in raw MammaPrint-index units over the cohort. No
  unverified high/low-risk categories are introduced.
- Joint: column 0 is a class logit and column 1 is an index. Sigmoid applies only
  to the class logit; the two outputs are never softmaxed together.
- Presence masks show concept share, not signed prediction attribution. Overlaps
  are averaged before shared per-concept quantile scaling; uncovered pixels are
  distinct from a covered tile with zero concept value.

## Doctor naming workflow and differences from IBD

Eight examples are selected by concept share, at most one per slide. Tiles below
the median total code, or with zero total code, are excluded. Shortages are
reported instead of duplicating slides or filling with ineligible examples.
Physical tile widths come from actual pyramid geometry (with nominal card MPP
clearly identified when the slide lacks it), not IBD's approximately 350 µm.
The source preprocessing selects each slide's scanner level by closest MPP;
the dataset card's level key is not necessarily that scanner index. Stored x/y
are relative to that selected tiling level. The report preserves that frame for
mask assembly, applies the original tile reader's downsample conversion, and
uses consistent level-relative ratios for viewport centering. Stored model
coordinate inputs remain unchanged, including for spatial transformers.

The HTML starts with labels/effects hidden. A reviewer can enter a concept name,
unclear/mixed flag, expected luminal type, same-as concept and comment. Revealing
results freezes expected-type responses. Copy answers exports the dictionary,
source-run and split identity; nothing is sent automatically. Reloading loses
answers, so copy them during the session. The blinding is a review convention,
not a security boundary; numerical content is present in the HTML.

The exact generator of Adam's deployed `naming_k16.html` was not available.
This page adapts his checked-in report plus the supplied Slack instructions. It
does not claim pixel-for-pixel parity. xOpat links center the tile, but the exact
original dashed-square annotation convention still needs his template. Do not
claim live viewer alignment has been checked until a VPN-enabled review.

Questions for Adam/the pathologist:

1. Can Adam share the actual naming-page generator and its source run/config?
2. Does an existing lab adapter already evaluate these interventions through
   arbitrary MIL heads/aggregators?
3. Is “expected luminal A/B / no effect / unsure” the desired blinded question,
   or should breast reviewers judge a different endpoint? The current wording
   is explicitly proposed; raw index effects are revealed separately.

## Local tests and current limits

```bash
uv run --frozen --extra explainability python -m unittest discover \
  -s tests -p 'test_concept_*.py' -v
```

The report tests also need Node for the real JavaScript handler checks and the
native OpenSlide/libvips libraries for TIFF readback. The synthetic end-to-end
test creates temporary RGB TIFFs, signed embeddings and a checkpoint, fits the
dictionary, evaluates it and reads the resulting masks/report. It never writes
to a training run or uses clinical patient data.

`fixture_mode=true` selects a <=4096-tile in-process smoke fit; resulting reports
are visibly marked FIXTURE and cannot be used for a normal report without an
explicit fixture-mode acknowledgement. Production uses the copied Ray streaming
iterator. For local files, set `local_source_config=/path/resolved.yaml`,
`local_checkpoint=/path/checkpoint.ckpt`, `training_sha=<verified SHA>` and
`log_to_mlflow=false`; the usual source provenance/model checks still apply.

No downloaded WSI, embedding parquet or checkpoint was found by a search
including ignored repository files at implementation time. Real-checkpoint
prediction parity, cohort fidelity, GPU memory/runtime and live xOpat alignment
must therefore be checked on the cluster after VPN access is ready. Test output
from a synthetic fixture does not substitute for those scientific checks.

See the [implementation verification and independent review](superpowers/reviews/2026-11-14-concept-explainability.md)
for completed checks, corrections and remaining clarification points.

## Bounded local real-asset pilot

For existing VPN-accessible assets without the original cluster mounts, use
[`run_local_concept_pilot.py`](../scripts/ml/run_local_concept_pilot.py) and the
[local pilot skill](../.agents/skills/mammaprint-concept-local-pilot/SKILL.md).
The default command prints a side-effect-free plan. Execution requires explicit
service URLs, source run and experiment identity, pinned cached source/label/
checkpoint assets, a private four-bag roster, outside-Git output/cache roots and
`--execute`. It never instantiates the full remote embedding directory.

The pilot calls the existing K16 fit/evaluate/report functions with two complete
training bags and two complete validation bags. The dictionary is explicitly a
training-subset fixture, with patient/case separation unverified. Original
train/val/test artifact **membership metadata** establish slide disjointness;
no synthetic test bag is introduced and no test bag is downloaded or scored.
Labels and geometry come from the pinned persisted tiling table, which does not
prove the bytes of the training label CSV.

At most two validation MRXS ZIPs are streamed through strict 512 MiB/archive,
1 GiB combined compressed caps (2/4 GiB expanded), with safe sidecar-preserving
extraction and OpenSlide/geometry checks. If complete WSIs exceed those bounds,
numerical smoke evaluation still runs; the manifest explicitly records that the
WSI report was skipped. All artifacts remain private and local, never logged to
the source run. A generated local report does not establish remote xOpat access
or live viewer alignment. The skill contains exact parameterized commands,
inputs, failure handling, cleanup and safe verifier instructions.
