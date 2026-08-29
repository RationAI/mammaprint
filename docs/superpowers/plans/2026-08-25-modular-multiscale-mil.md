# Modular Multiscale MIL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lazily assemble complete, non-overlapping aligned embedding pyramids and provide a tightly typed, vectorized, plug-and-play multiscale MIL pipeline with documented null/mean/attention/Transformer alternatives and reproducible experiment presets.

**Architecture:** Reuse existing source resolution, labels, collation, identity encoder, pooling primitives, LinearHead, MLPHead, losses, and Lightning module. Generalize the existing `feat/multiscale` prototype into a staged aggregator whose validated representation descriptor tracks tokens, level vectors, pyramid vectors, structured slide outputs, and flat slide vectors. Keep the single-scale API unchanged.

**Tech Stack:** Python 3.12, PyTorch, Lightning, Hydra/OmegaConf, pandas/Parquet, unittest, mypy, Ruff, and the repository's MLflow integration.

**Spec:** `docs/superpowers/specs/2026-08-25-modular-multiscale-mil-design.md`

**Backlog:** `docs/superpowers/backlog/2026-08-25-modular-multiscale-mil-backlog.md`

## Global Constraints

- Do not retile, re-embed, or create a persistent multiscale embedding artifact.
- Retain only complete pyramids; do not add padding or missing-token behavior.
- Configure `tile_extent`; validate all level cards against it; never hardcode 224.
- Preserve deterministic L5/L4/L3 and row-major token order.
- Default to deterministic non-overlapping anchors; retain `all` as an ablation.
- Return ordered anchor coordinates in metadata, but do not feed them to the model.
- Keep all single-scale config paths and runtime behavior unchanged.
- Treat YAML `null` as identity; never silently aggregate, flatten, pad, or drop axes.
- Reject invalid stage combinations before the first training batch.
- Vectorize over all pyramids `R`; do not loop over pyramids in learned modules.
- Work test-first and commit after every task.

---

## Task 1: Integrate the proven lazy-pyramid foundation

**Files:**

- Integrate commits: `a249f48`, `476fc14`, `72e495b`, `0121168`, `c4e5525`
- Verify: `ml/typing.py`
- Verify: `ml/data/datasets/pyramid.py`
- Verify: `ml/models/aggregators/multiscale.py`
- Verify: `configs/data/_pyramid_template.yaml`
- Verify: `configs/data/pyramid/tissue_only_l5_l4_l3.yaml`
- Verify: `configs/experiment/ml/train_mil_embeddings_multiscale.yaml`
- Verify: multiscale tests introduced by those commits

- [ ] Use `superpowers:using-git-worktrees` to create an isolated feature worktree
  from current `feat/tiling-values`.

- [ ] Cherry-pick the existing tested implementation, excluding its superseded
  planning-only commits:

```bash
git cherry-pick a249f48 476fc14 72e495b 0121168 c4e5525
```

Expected: five commits apply. On conflict, retain the current branch's newer data
cards/cancer-mask changes while adding the multiscale files.

- [ ] Verify the foundation:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.data.test_pyramid_layout tests.data.test_pyramid_dataset \
  tests.models.test_multiscale_aggregator tests.config.test_multiscale_config
uv run mypy ml/typing.py ml/data/datasets/pyramid.py \
  ml/models/aggregators/multiscale.py ml/models/module.py ml/data/datamodule.py
uv run ruff check ml/typing.py ml/data/datasets/pyramid.py \
  ml/models/aggregators/multiscale.py tests/data tests/models tests/config
```

Expected: all commands exit 0. Conflict-free cherry-picks already provide
task-sized commits, so do not create a redundant integration commit.

## Task 2: Add typed anchor metadata, non-overlap selection, and audit

**Files:**

- Modify: `ml/typing.py`
- Modify: `ml/data/datasets/pyramid.py`
- Modify: `ml/data/datasets/__init__.py`
- Modify: `ml/data/__init__.py`
- Modify: `configs/data/_pyramid_template.yaml`
- Modify: `configs/data/pyramid/tissue_only_l5_l4_l3.yaml`
- Modify: `tests/data/test_pyramid_dataset.py`
- Create: `tests/data/test_pyramid_audit.py`

- [ ] Add failing tests proving that two returned anchors have the same order as
  `bag[0]`, `bag[1]`; that `non_overlapping` removes intersecting L5 footprints;
  that boundary-touching footprints remain; that `all` retains both; and that the
  same `selection_seed` gives the same result twice.

- [ ] Add failing audit tests with complete, incomplete, overlap-rejected, and
  zero-retained slides. Assert exact candidate, pre-thinning complete, retained,
  incomplete, missing-by-level, completion-rate, overlap-retention-rate, and
  rejected values.

- [ ] Add these types without weakening existing `SlideMetadata`:

```python
class PyramidSlideMetadata(TypedDict):
    slide_id: str
    anchors: tuple[Coordinate, ...]

class OverlapPolicy(StrEnum):
    NON_OVERLAPPING = "non_overlapping"
    ALL = "all"

@dataclass(frozen=True, slots=True)
class PyramidSelectionAudit:
    slide_id: str
    split: str
    label: str
    candidates: int
    complete_before_thinning: int
    retained: int
    incomplete: int
    missing_by_level: Mapping[int, int]
    completion_rate: float
    overlap_retention_rate: float
    rejected: bool
```

- [ ] Implement stable slide-specific candidate ordering with a local
  `random.Random(f"{selection_seed}:{slide_id}")`. Accept a complete candidate
  only if its rectangle `(x, y, x+tile_extent, y+tile_extent)` has no positive-area
  intersection with accepted rectangles. Do not use process-global RNG state.

- [ ] Expose typed `audit_records`, `audit_frame()`, and `audit_summary()`.
  Summary groups by split and label and reports slide/rejected counts, totals,
  completion and overlap-retention rates, and retained p25/p50/p75.

- [ ] Add dataset config values:

```yaml
pyramid:
  require_complete: true
  overlap_policy: non_overlapping
  selection_seed: 0
```

Reject `require_complete=false`; it documents the invariant but does not activate
an unsupported path.

- [ ] Run tests/checks:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.data.test_pyramid_layout tests.data.test_pyramid_dataset \
  tests.data.test_pyramid_audit
uv run mypy ml/typing.py ml/data/datasets/pyramid.py
uv run ruff check ml/typing.py ml/data/datasets/pyramid.py tests/data
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add ml/typing.py ml/data configs/data tests/data
git commit -m "feat: audit complete multiscale pyramid selection"
```

## Task 3: Define and validate representation transitions

**Files:**

- Create: `ml/models/aggregators/multiscale/__init__.py`
- Create: `ml/models/aggregators/multiscale/spec.py`
- Move/refactor: `ml/models/aggregators/multiscale.py` into the package
- Modify: `ml/models/aggregators/__init__.py`
- Create: `tests/models/multiscale/test_pipeline_spec.py`

- [ ] Add failing table tests for these output dimensions:

```text
null -> null -> null -> mean -> flatten           = 21 * D
null -> mean -> concat -> mean -> null             = 3 * D
shared_linear -> mean -> mean -> mean -> null      = H
shared_linear -> shared_transformer -> concat
              -> gated_attention -> null           = 3 * H
shared_linear -> null -> joint_transformer
              -> gated_attention -> null           = H
```

- [ ] Add failing tests for token input with concat/mean/gated-sum, level input
  with flatten/joint-Transformer, structured input with gated slide attention,
  ordinary training with null slide pool, flatten before removing `R`, and
  head-width mismatch.

- [ ] Implement exact enum/literal stage names, where Python `None` is the only
  null representation:

```python
type FeatureAdapterName = Literal["shared_linear"] | None
type WithinLevelName = Literal["mean", "max", "gated_attention", "shared_transformer"] | None
type LevelFusionName = Literal["flatten", "concat", "mean", "gated_sum", "joint_transformer"] | None
type SlidePoolName = Literal["mean", "max", "gated_attention"] | None
type FinalizerName = Literal["flatten"] | None
```

- [ ] Implement frozen `Representation` and `MultiScalePipelineSpec` dataclasses.
  `transitions()` returns every semantic state; `out_dim` returns an integer only
  for final `SlideVector`; `validate_head_dim()` reports both dimensions and the
  full rejected spec. Kinds are `pyramid_tokens`, `level_vectors`,
  `pyramid_vectors`, `token_slide`, `level_slide`, and `slide_vector`.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.models.multiscale.test_pipeline_spec
uv run mypy ml/models/aggregators/multiscale/spec.py
uv run ruff check ml/models/aggregators/multiscale tests/models/multiscale
git add ml/models/aggregators tests/models/multiscale
git commit -m "refactor: type multiscale representation transitions"
```

Expected: tests and checks exit 0 before commit.

## Task 4: Add projection and vectorized within-level primitives

**Files:**

- Create: `ml/models/aggregators/multiscale/adapter.py`
- Create: `ml/models/aggregators/multiscale/within_level.py`
- Modify: `ml/models/aggregators/attention.py`
- Modify: `ml/models/aggregators/transformer.py`
- Modify: existing attention/Transformer tests
- Create: `tests/models/multiscale/test_feature_adapter.py`
- Create: `tests/models/multiscale/test_within_level.py`

- [ ] Add failing equivalence tests proving that, in eval mode with dropout zero,
  `forward_batched(x.unsqueeze(0))` equals existing rank-2 `forward(x)` for
  `AttentionMIL` and `TransformerMIL`. Assert `(B,D)` pooled output, `(B,N)`
  attention, and per-bag weights summing to one.

- [ ] Refactor both existing aggregators without changing public behavior:

```python
def forward_batched(self, bags: Tensor) -> tuple[Tensor, Tensor]:
    """Pool (B, N, D) to (B, D), returning weights (B, N)."""

def forward(self, bag: Tensor) -> tuple[Tensor, Tensor]:
    pooled, attention = self.forward_batched(bag.unsqueeze(0))
    return pooled[0], attention[0]
```

Preserve state-dict names unless a migration test justifies a change.

- [ ] Add failing `SharedLinearAdapter` tests. It must have three distinct
  LayerNorms and one shared Linear, transform `(R,21,2560)` to `(R,21,512)`,
  preserve token order, and propagate gradients. Null adapter returns the exact
  input object.

- [ ] Implement level-slice derivation from `level_ratios`; apply per-level
  LayerNorm, shared Linear, GELU, and dropout, then concatenate in original order.

- [ ] Add hand-computable tests for null, mean, and max within-level operations.
  Add shape/normalization/gradient tests for gated attention and shared Transformer.
  Patch the learned module to assert exactly three batched calls—one each for
  `(R,1,D)`, `(R,4,D)`, `(R,16,D)`—and no calls per pyramid.

- [ ] Implement `SharedTransformerWithinLevel` with one shared one-layer
  `TransformerMIL`, separate learned relative-position tables of lengths 1/4/16,
  and one learned three-entry level table. Add position plus level context only in
  Transformer modes; mean/max/gated attention do not receive these embeddings.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.models.test_attention_aggregator \
  tests.models.test_transformer_aggregator \
  tests.models.multiscale.test_feature_adapter \
  tests.models.multiscale.test_within_level
uv run mypy ml/models/aggregators/attention.py \
  ml/models/aggregators/transformer.py ml/models/aggregators/multiscale
uv run ruff check ml/models/aggregators tests/models
git add ml/models/aggregators tests/models
git commit -m "feat: vectorize balanced within-level pyramid processing"
```

Expected: tests and checks exit 0 before commit.

## Task 5: Implement fusion, slide pooling, finalization, and pipeline

**Files:**

- Create: `ml/models/aggregators/multiscale/fusion.py`
- Create: `ml/models/aggregators/multiscale/pipeline.py`
- Modify: `ml/models/aggregators/multiscale/__init__.py`
- Modify: `ml/models/aggregators/__init__.py`
- Create: `tests/models/multiscale/test_fusion.py`
- Create: `tests/models/multiscale/test_pipeline.py`
- Modify: `tests/models/test_multiscale_aggregator.py`

- [ ] Add failing fusion tests:

  - null preserves object and semantic axes;
  - flatten returns `(R,21D)` in token-major order;
  - concat returns `(R,3D)` in L5/L4/L3 order;
  - mean is the exact equal mean of three levels;
  - gated sum returns `(R,D)` and normalized three-level weights;
  - joint Transformer returns `(R,D)` through one batched call.

- [ ] Use focused modules only for trainable operations. Null, reshape, concat,
  mean, and max must not introduce trainable placeholder parameters.

- [ ] Add failing end-to-end pipeline tests for:

```text
raw tokens + null fusion + slide mean + final flatten -> (21D,), no attention
raw level mean + concat + slide mean                  -> (3D,), no attention
project + level mean + concat + slide gate            -> (3H,), attention (R,)
project + shared transformer + concat + slide gate    -> (3H,), attention (R,)
project + null within + joint transformer + gate      -> (H,), attention (R,)
feature export + null slide pool                      -> retains R
```

- [ ] Implement `ModularMultiScaleMIL(Aggregator[PyramidBag])`. Its constructor
  builds a validated `MultiScalePipelineSpec`, creates only selected modules,
  exposes `out_dim`, validates `(R,K,D)`, and executes stages in the documented
  order. Mean/max slide pooling reduce only dimension zero and preserve fixed
  axes. Gated slide pooling accepts rank-2 pyramid vectors only. Final flatten is
  allowed only after `R` is removed.

- [ ] Retain `MultiScaleMIL` as a deprecated compatibility wrapper for the initial
  joint-21-token Transformer plus gated slide attention. New configs target
  `ModularMultiScaleMIL`; do not break the already-started prototype config.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.models.multiscale.test_fusion tests.models.multiscale.test_pipeline \
  tests.models.test_multiscale_aggregator
uv run mypy ml/models/aggregators/multiscale ml/models/aggregators/__init__.py
uv run ruff check ml/models/aggregators/multiscale tests/models/multiscale
git add ml/models/aggregators tests/models
git commit -m "feat: compose modular multiscale MIL stages"
```

Expected: tests and checks exit 0 before commit.

## Task 6: Preserve tight sample types and prediction metadata end to end

**Files:**

- Modify: `ml/typing.py`
- Modify: `ml/data/datamodule.py`
- Modify: `ml/models/module.py`
- Modify/create: `tests/models/test_module.py`
- Modify: `tests/type_checks/invalid_pyramid_pair.py`
- Create: `tests/type_checks/test_invalid_pyramid_pair.py`

- [ ] Add a failing runtime test that a pyramid sample passes through identity
  encoder, modular aggregator, and existing LinearHead. Assert `predict_step`
  returns anchors unchanged and in slide-attention order.

- [ ] Add a negative mypy fixture pairing a pyramid sample with a single-scale-only
  aggregator. Invoke mypy in the test and assert its incompatible-type error.

- [ ] Parameterize the shared sample type without falling back to `Any`:

```python
type SampleOf[BagT: Tensor, MetadataT] = tuple[BagT, Tensor, MetadataT]
type MILSample = SampleOf[Bag, SlideMetadata]
type PyramidSample = SampleOf[PyramidBag, PyramidSlideMetadata]
type Sample = MILSample | PyramidSample
```

Use generic unpack/collate helpers. If Lightning forces one dynamic boundary,
contain one annotated cast at that boundary and explain it locally.

- [ ] Validate aggregator/head dimensions in `MammaprintModule.__init__`:

```python
if aggregator.out_dim != head.in_dim:
    raise ValueError(
        f"Aggregator output width {aggregator.out_dim} does not match "
        f"head input width {head.in_dim}."
    )
```

Feature export stays an explicit aggregator method and does not weaken training's
flat-head requirement.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.models.test_module tests.type_checks.test_invalid_pyramid_pair
uv run mypy ml/typing.py ml/data/datamodule.py ml/models/module.py
uv run ruff check ml/typing.py ml/data/datamodule.py ml/models/module.py \
  tests/models/test_module.py tests/type_checks
git add ml/typing.py ml/data/datamodule.py ml/models/module.py \
  tests/models/test_module.py tests/type_checks
git commit -m "refactor: preserve multiscale types through prediction"
```

Expected: checks exit 0; the negative-fixture test passes because mypy rejects
the intentionally invalid program.

## Task 7: Add the complete Hydra surface and parameter documentation

**Files:**

- Create: `configs/ml/aggregator/multiscale_modular.yaml`
- Create: `configs/ml/aggregator/multiscale_presets/*.yaml`
- Modify: `configs/data/_pyramid_template.yaml`
- Modify: `configs/data/pyramid/tissue_only_l5_l4_l3.yaml`
- Create: `docs/multiscale-mil-config.md`
- Modify: `tests/config/test_multiscale_config.py`

- [ ] Add failing composition/instantiation tests for every named preset. Assert
  exact stage enum values and `aggregator.out_dim == head.in_dim`.

- [ ] Add failing overrides for each YAML null and prove Hydra passes Python
  `None`, not a string. Assert invalid combinations produce descriptive errors.

- [ ] Create the base config with every stage visible:

```yaml
_target_: ml.models.aggregators.ModularMultiScaleMIL
feature_adapter: null
within_level: mean
level_fusion: concat
slide_pool: mean
finalizer: null
feature_dim: ${feature_dim}
hidden_dim: ${multiscale_hidden_dim}
level_ratios: ${pyramid.level_ratios}
projection_dropout: 0.1
attention_dim: 256
transformer_num_layers: 1
transformer_num_heads: 8
transformer_mlp_ratio: 4.0
transformer_dropout: 0.1
position_embedding: null
level_embedding: null
feature_export: false
```

Transformer presets override both embedding values to `learned`. Non-Transformer
presets keep them null. Sinusoidal and no-position Transformer comparisons remain
backlog, not accepted-but-unimplemented strings.

- [ ] Create these presets, spelling out all five stages even when null:

```text
multiscale_mean_linear
multiscale_level_mean_linear
multiscale_raw_flat_linear
multiscale_mean_mlp
multiscale_projected_mean
multiscale_slide_attention
multiscale_local_attention
multiscale_transformer_mean_slide
multiscale_transformer
multiscale_joint_transformer
```

Each experiment defines top-level `slide_dim` and binds `ml.head.in_dim` to it;
the runtime equality check remains authoritative.

- [ ] Document every allowed value, input/output shape, parameterization, and use
  case in `docs/multiscale-mil-config.md`. Include null restrictions, flatten vs
  concat, learned position/level examples, vectorized shapes, valid/invalid YAML,
  output widths for `D=2560,H=512`, and why concat—not level gating—is default.
  Link the design, plan, and backlog.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.config.test_multiscale_config
uv run ruff check tests/config/test_multiscale_config.py
git add configs/ml/aggregator configs/data docs/multiscale-mil-config.md \
  tests/config/test_multiscale_config.py
git commit -m "feat: configure modular multiscale MIL experiments"
```

Expected: every preset composes/instantiates; invalid overrides fail as asserted.

## Task 8: Add experiment configs and selection-audit artifacts

**Files:**

- Create: `configs/experiment/ml/multiscale/*.yaml`
- Create: `ml/logging/pyramid_audit.py`
- Modify: `train.py`
- Create: `tests/config/test_multiscale_experiments.py`
- Create: `tests/logging_tests/test_pyramid_audit.py`

- [ ] Add failing tests composing all ten experiments with the existing L5/L4/L3
  cards. Assert common splits, loss, optimizer, scheduler, metrics,
  `test_after_fit=false`, and explicit seed.

- [ ] Give each config a precise run name and resolved metadata containing tile
  extent, overlap policy/seed, five stages, feature/hidden/slide widths, head,
  loss, trainable parameter count, and training seed.

- [ ] Add failing fake-logger tests, then implement an audit logger that accepts a
  dataset exposing `audit_frame()`/`audit_summary()`, writes CSV and JSON into the
  managed run output, and logs them with existing MLflow integration. It is a
  no-op for single-scale data; datasets themselves never call MLflow.

- [ ] Run and commit:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest \
  tests.config.test_multiscale_experiments tests.logging_tests.test_pyramid_audit
uv run mypy ml/logging/pyramid_audit.py train.py
uv run ruff check ml/logging/pyramid_audit.py train.py \
  tests/config/test_multiscale_experiments.py tests/logging_tests/test_pyramid_audit.py
git add configs/experiment/ml/multiscale ml/logging/pyramid_audit.py train.py \
  tests/config/test_multiscale_experiments.py tests/logging_tests/test_pyramid_audit.py
git commit -m "feat: make multiscale experiments auditable"
```

Expected: all commands exit 0.

## Task 9: Verify implementation and reconcile documentation

**Files:**

- Modify: `README.md`
- Modify if needed: `docs/multiscale-mil-config.md`
- Modify if needed: design and backlog documents linked from this plan

- [ ] Add one README link and minimal multiscale training command; do not duplicate
  the full config reference.

- [ ] Run the complete suite and repository checks:

```bash
MPLCONFIGDIR=/tmp/mammaprint-mpl uv run python -m unittest discover -s tests
uv run mypy ml
uv run ruff check .
uv run ruff format --check .
```

Expected: all commands exit 0.

- [ ] Compose every experiment without training:

```bash
for preset in multiscale_mean_linear multiscale_level_mean_linear \
  multiscale_raw_flat_linear multiscale_mean_mlp \
  multiscale_projected_mean multiscale_slide_attention \
  multiscale_local_attention multiscale_transformer_mean_slide \
  multiscale_transformer multiscale_joint_transformer
do
  uv run python train.py --cfg job experiment=ml/multiscale/$preset >/dev/null || exit 1
done
```

Expected: loop exits 0.

- [ ] Run a two-slide, one-epoch CPU smoke fit for `multiscale_mean_linear` and
  `multiscale_transformer`. Assert finite loss and ordered prediction anchors.

- [ ] Audit documentation/config/code vocabulary:

```bash
rg -n "feature_adapter|within_level|level_fusion|slide_pool|finalizer" \
  docs/multiscale-mil-config.md configs/ml/aggregator \
  ml/models/aggregators/multiscale
rg -n "TODO|TBD|placeholder|implement later" \
  docs/multiscale-mil-config.md \
  docs/superpowers/specs/2026-08-25-modular-multiscale-mil-design.md
```

Expected: stage names agree; the second command returns no unfinished production
documentation. Future-facing backlog text is intentionally excluded.

- [ ] Commit:

```bash
git add README.md docs
git commit -m "docs: explain modular multiscale MIL configuration"
```

## Task 10: Run the initial experiment ladder

**Artifacts:**

- Record: MLflow runs in the existing MammaPrint experiment
- Create: `docs/experiments/multiscale-initial-results.md`

- [ ] Freeze and record split identifiers. Use screening seeds `17`, `29`, `43`.
  Keep test locked throughout selection.

- [ ] Establish same-protocol single-scale controls: L3, L4, and L5 with mean plus
  LinearHead; include the best existing attention run only if splits/protocol match.

- [ ] Run design-spec orders 1–5. Record resolved config/commit, trainable parameter
  count, audit by split/label, best/final validation metric, wall time, peak H100
  memory, and seed.

- [ ] Apply gate 1: continue to orders 6–10 only if a simple multiscale preset
  improves consistently over the strongest single-scale control or exposes a
  specific testable underfitting pattern. One favorable seed is insufficient.

- [ ] Run `overlap_policy=all` for `multiscale_mean_linear` and the best learned
  architecture only. Compare metrics and retained-pyramid distributions.

- [ ] Select the strongest simple model, strongest learned model, and best overlap
  policy. Re-run finalists with seeds `17`, `29`, `43`, `71`, `97`; report mean,
  standard deviation, individual values, parameters, and runtime.

- [ ] Write the results document with keep/drop decisions for every optional stage.
  Only after selection is frozen, run one locked test evaluation per finalist.

- [ ] Commit results, excluding checkpoints and downloaded artifacts:

```bash
git add docs/experiments/multiscale-initial-results.md
git commit -m "docs: report initial multiscale MIL experiments"
```

## Experiment Interpretation Rules

- Universal: complete aligned pyramids, overlap policy/audit, typed anchors,
  vectorized execution, and modular representation validation.
- Interchangeable: projection, within-level aggregation, fusion, slide pooling,
  finalizer, head, and loss, subject to shape compatibility.
- Change one concept at a time; do not add Transformer, slide attention, MLP, and
  overlap changes in one comparison.
- H100 memory makes these models feasible but does not increase the roughly 2,000
  independent slides. Prefer the lowest-capacity competitive model.
- Promote backlog work only when this ladder identifies the deficiency it tests.
