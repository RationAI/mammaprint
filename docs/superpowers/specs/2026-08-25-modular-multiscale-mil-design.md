# Modular Multiscale MIL Design

## Decision summary

The multiscale path will lazily assemble complete, spatially aligned level-5,
level-4, and level-3 pyramids from the existing embedding Parquets. It will not
retile slides, recompute embeddings, or persist a second multiscale embedding
dataset. Each slide becomes a variable number `R` of complete pyramids, and each
pyramid contains 21 tokens in a fixed order:

```text
level 5:  1 token   (1 x 1)  -> token 0
level 4:  4 tokens  (2 x 2)  -> tokens 1..4, row-major
level 3: 16 tokens  (4 x 4)  -> tokens 5..20, row-major
```

The model will be a typed pipeline of independently configurable stages:

```text
PyramidBag (R, 21, D)
  -> feature_adapter
  -> within_level
  -> level_fusion
  -> slide_pool
  -> finalizer
  -> existing LinearHead or MLPHead
  -> existing task loss
```

Every stage is present in configuration. YAML `null` means that the stage is an
identity and preserves its current representation. Invalid combinations fail at
model construction; the implementation never silently flattens, averages, pads,
or drops an axis.

## Pyramid geometry

The coarsest configured level is the anchor. For anchor coordinate `(x, y)`,
level tile extent `T`, and integer ratio

```text
r(level) = anchor_mpp / level_mpp
```

the required coordinates at that level are:

```text
(r*x + column*T, r*y + row*T)
for row in range(r)
for column in range(r)
```

For ratios `(1, 2, 4)`, an anchor at grid coordinate `(3, 3)` uses:

```text
level 5: (3, 3)
level 4: (6, 6), (7, 6), (6, 7), (7, 7)
level 3: (12..15, 12..15), row-major
```

Stored Parquet coordinates are pixel coordinates, so the implementation applies
the same formula with offsets of `T`. `tile_extent` is a positive configuration
value and is never hardcoded to 224. All selected level cards must declare the
same extent. MPP ratios must be positive integers within numerical tolerance.

## Completeness, overlap, and selection audit

Only complete pyramids are retained. An anchor is incomplete if any of its 21
coordinates is absent. Slides with zero retained pyramids are excluded.

`overlap_policy` has two values:

- `non_overlapping` is the default. Complete anchors are considered in a stable,
  slide-specific deterministic order. A candidate is retained only when its
  level-5 footprint does not overlap a previously retained footprint.
- `all` retains every complete anchor and is an ablation for measuring the effect
  of redundant, overlapping views.

The default reduces repeated fine-level embeddings and stops densely tiled areas
from dominating only because they generate many nearly identical pyramids. The
algorithm is deterministic for a fixed slide and seed.

The dataset exposes typed per-slide audit records containing:

- candidate anchors;
- complete anchors before overlap thinning;
- retained anchors;
- incomplete anchors;
- missing-anchor counts by level;
- completion rate;
- overlap-retention rate;
- whether the slide was rejected.

It also produces aggregate tables by split and label with slide counts, rejected
slide counts, totals, rates, and retained-pyramid distribution quantiles. Training
logs these tables as CSV and JSON artifacts. This audits selection bias without
changing the user's already-created physical alignment.

## Prediction metadata and global coordinates

The sample metadata retains ordered level-5 anchor coordinates, aligned exactly
with the first `R` dimension of the returned `PyramidBag` and any slide-attention
weights. Coordinates are not model inputs in this implementation.

Preserving them now means a prediction can be mapped back to its source regions,
compared with the completeness audit, and used by a future global spatial model.
It does not mean adding a slide-coordinate positional encoding today.

## Typed representations

PyTorch cannot statically encode tensor dimensions, so the implementation combines
branded tensor aliases with a validated representation descriptor. The following
semantic shapes are distinct even though their runtime value is a `Tensor`:

| Representation | Shape | Meaning |
|---|---:|---|
| `PyramidTokens` | `(R, 21, D)` | All fixed-layout tokens per pyramid |
| `LevelVectors` | `(R, 3, D)` | One vector for each of L5, L4, L3 |
| `PyramidVectors` | `(R, P)` | One flat vector per pyramid |
| `TokenSlide` | `(21, D)` | Slide-pooled, still token-structured |
| `LevelSlide` | `(3, D)` | Slide-pooled, still level-structured |
| `SlideVector` | `(P,)` | Valid input to the existing heads |

A frozen `MultiScalePipelineSpec` validates the stage transitions and calculates
the final dimension. The runtime pipeline validates ranks and widths at its
boundary. The Hydra boundary remains necessarily dynamic, but unsupported stage
combinations fail before the first training batch.

Single-scale `Bag`, `MILSample`, datasets, aggregators, and configs remain valid
and unchanged. Shared primitives gain batched methods without changing their
existing single-bag `forward` behavior.

## Stage configuration reference

### `feature_adapter`

This stage operates on the 2560-dimensional pretrained tile embeddings.

| Value | Operation | Output width | Intended use |
|---|---|---:|---|
| `null` | Identity | `feature_dim` | Parameter-light raw-embedding baselines |
| `shared_linear` | Level-specific LayerNorm, one Linear shared by all levels, GELU, dropout | `hidden_dim` | Default learned bottleneck |

`shared_linear` uses separate normalization statistics/affine parameters for L5,
L4, and L3, but the same projection matrix. With `feature_dim=2560` and
`hidden_dim=512`, it reduces later attention and MLP cost while encouraging a
common cross-level space.

`per_level_linear` is deliberately in the backlog. It uses about three times the
projection parameters and should be tested only if the shared projection appears
to underfit.

### `within_level`

This stage works independently inside the 1-, 4-, and 16-token level groups.

| Value | Operation | Result |
|---|---|---|
| `null` | Preserve every token | `PyramidTokens (R,21,D)` |
| `mean` | Equal average inside each level | `LevelVectors (R,3,D)` |
| `max` | Elementwise maximum inside each level | `LevelVectors (R,3,D)` |
| `gated_attention` | Learned gated weighted sum inside each level | `LevelVectors (R,3,D)` |
| `shared_transformer` | One shared local Transformer, called once per level as a batch | `LevelVectors (R,3,D)` |

The one-token L5 group passes through mean/max unchanged. Gated attention has no
meaningful choice for L5 but uses the same interface. The shared Transformer is
one layer by default (`hidden_dim=512`, 8 heads, MLP ratio 4, dropout 0.1). It is
plausible for 2,000 slides because each slide yields many supervised pyramid
views internally, but it is not the first experiment: simpler baselines determine
whether spatial organization itself provides signal.

For `shared_transformer`, each token receives:

```text
projected_embedding + learned_relative_position + learned_level_embedding
```

Learned relative positions are small tables for the fixed slots: one L5 slot,
four L4 slots, and sixteen L3 slots. For example, the upper-left L3 tile and
lower-right L3 tile receive different trainable vectors. Learned level embeddings
are three trainable vectors—one each for L5, L4, and L3—so the shared Transformer
can distinguish magnification even when image features look similar. With width
512, these tables contain only `(21 + 3) * 512 = 12,288` parameters. They are
trained end-to-end and are not slide IDs, so 2,000 slides do not require 2,000
separate embeddings.

Learned position and level embeddings are used only by Transformer-based local
processing. Mean, max, gated attention, LinearHead, and MLPHead do not need them.

Vectorized processing means each level is processed for all `R` pyramids in one
batched call:

```text
L5: (R,  1, D) -> shared local module -> (R, D)
L4: (R,  4, D) -> shared local module -> (R, D)
L3: (R, 16, D) -> shared local module -> (R, D)
```

There is no Python loop over pyramids. Existing attention and Transformer pools
gain `forward_batched`; their current rank-2 `forward` delegates to that method.

### `level_fusion`

This stage combines or preserves the levels within each pyramid.

| Value | Accepted input | Operation | Result |
|---|---|---|---|
| `null` | tokens or level vectors | Preserve axes | Same representation |
| `flatten` | `PyramidTokens` | Put all 21 token vectors behind each other | `PyramidVectors (R,21D)` |
| `concat` | `LevelVectors` | Put L5, L4, L3 vectors behind each other | `PyramidVectors (R,3D)` |
| `mean` | `LevelVectors` | Equal average of the three level vectors | `PyramidVectors (R,D)` |
| `gated_sum` | `LevelVectors` | Learn three normalized scalar weights per pyramid and sum | `PyramidVectors (R,D)` |
| `joint_transformer` | `PyramidTokens` | One Transformer over all 21 fixed tokens | `PyramidVectors (R,D)` |

`concat` is the recommended learned-model default after one-vector-per-level
processing. With exactly three known levels, concatenation preserves all three
vectors, gives each level equal dimensional capacity, and lets the head learn how
to combine them. It does not triple every model component; it triples only the
input width of the next flat operation.

`gated_sum` is optional, not default. It is lossy because three vectors become one
and should only be tested as a regularized alternative. `mean` is the parameter-free
version of that compression. `joint_transformer` is the direct evolution of the
existing prototype and provides a comparison with the balanced per-level design.

`flatten` differs from `concat`: flatten is for all 21 unaggregated tokens, while
concat is for the three already-aggregated level vectors. The validator rejects
using the wrong name for the current representation.

### `slide_pool`

This stage reduces the variable number of pyramids `R`.

| Value | Operation | Structured input support |
|---|---|---|
| `null` | Preserve `R` | Any; only feature-export or a future bag-aware head |
| `mean` | Equal average across pyramids | Flat, level-structured, or token-structured |
| `max` | Elementwise maximum across pyramids | Flat, level-structured, or token-structured |
| `gated_attention` | Learned scalar weight per pyramid and weighted sum | Flat `PyramidVectors` only |

Gated attention is useful here because `R` is variable and can be large. Unlike
gating across exactly three levels, slide attention selects among tissue regions
and returns an interpretable weight aligned with anchor metadata.

### `finalizer`

| Value | Operation | Use |
|---|---|---|
| `null` | Preserve the representation | Already-flat slide vectors or feature export |
| `flatten` | Reshape fixed remaining axes into one vector | Feed token/level-structured slide output to Linear/MLP head |

Flatten performs no pooling and has no trainable parameters. For example,
`within_level=null`, `level_fusion=null`, `slide_pool=mean`, and
`finalizer=flatten` turns `(R,21,2560)` into `(53760,)` after averaging only over
`R`. This is a valid raw-layout LinearHead baseline.

### Head and loss

The existing `LinearHead` and two-layer `MLPHead` are reused. Head choice is
independent once the pipeline produces `SlideVector` and its configured `in_dim`
matches the pipeline's calculated output dimension.

Existing BCE, regression, and joint losses remain interchangeable with either
head. The multiscale feature architecture does not require a new loss.

## Compatibility rules

The model constructor enforces these rules:

1. `within_level=null` produces tokens. It supports `level_fusion=null`,
   `flatten`, or `joint_transformer`, but not `concat`, `mean`, or `gated_sum`.
2. Non-null within-level pooling produces three level vectors. It supports
   `level_fusion=null`, `concat`, `mean`, or `gated_sum`, but not `flatten` or
   `joint_transformer`.
3. `slide_pool=gated_attention` requires a flat `PyramidVectors` input.
4. `slide_pool=mean|max` may preserve fixed token or level axes.
5. A normal Linear/MLP prediction run must remove `R` and end in `SlideVector`.
6. `slide_pool=null` is valid only for explicit feature-export mode or a future
   bag-aware head. It is rejected for ordinary training/evaluation.
7. `finalizer=flatten` is valid only after `R` has been removed.
8. The configured head `in_dim` must equal the calculated pipeline `out_dim`.

## Recommended defaults

The default dataset behavior is:

```yaml
pyramid:
  anchor_level: 5
  level_order: [5, 4, 3]
  level_ratios: [1, 2, 4]
  tile_extent: 224
  require_complete: true
  overlap_policy: non_overlapping
  selection_seed: 0
```

`tile_extent=224` is only the current data-card value and remains overrideable.
`require_complete` is fixed to `true` in this implementation.

The recommended learned model—not the first baseline—is:

```yaml
multiscale:
  feature_adapter: shared_linear
  within_level: shared_transformer
  level_fusion: concat
  slide_pool: gated_attention
  finalizer: null
  feature_dim: 2560
  hidden_dim: 512
  projection_dropout: 0.1
  transformer:
    num_layers: 1
    num_heads: 8
    mlp_ratio: 4.0
    dropout: 0.1
    position_embedding: learned
    level_embedding: learned
  attention_dim: 256
```

It produces a 1536-dimensional slide vector for either existing head. H100 memory
is ample for this model; the limiting factor is statistical power from roughly
2,000 slides, so the experiment sequence starts with lower-capacity controls.

## Initial experiment sequence

All experiments use the same split definitions, label mode, optimizer policy,
early-stopping rule, and evaluation metric. Screen with three fixed seeds. Re-run
the selected simple baseline, best learned model, and best overlap ablation with
five seeds. Keep the test split locked (`test_after_fit=false`) until selection is
complete.

| Order | Preset | Adapter | Within level | Fusion | Slide pool | Finalizer | Head | Question |
|---:|---|---|---|---|---|---|---|---|
| 0 | existing L3/L4/L5 baselines | n/a | n/a | n/a | mean/attention | n/a | linear | Which single scale is strongest? |
| 1 | `multiscale_mean_linear` | null | mean | concat | mean | null | linear | Does aligned multiscale data help without learned aggregation? |
| 2 | `multiscale_level_mean_linear` | null | mean | mean | mean | null | linear | Is equal level averaging enough? |
| 3 | `multiscale_raw_flat_linear` | null | null | null | mean | flatten | linear | Does fixed 21-slot layout itself carry signal? |
| 4 | `multiscale_mean_mlp` | null | mean | concat | mean | null | MLP | Does nonlinear cross-level fusion help? |
| 5 | `multiscale_projected_mean` | shared_linear | mean | concat | mean | null | linear | Does the projection bottleneck improve generalization? |
| 6 | `multiscale_slide_attention` | shared_linear | mean | concat | gated_attention | null | linear | Is learned region selection useful? |
| 7 | `multiscale_local_attention` | shared_linear | gated_attention | concat | gated_attention | null | linear | Is within-level learned pooling useful? |
| 8 | `multiscale_transformer_mean_slide` | shared_linear | shared_transformer | concat | mean | null | linear | Does local geometry help without slide attention? |
| 9 | `multiscale_transformer` | shared_linear | shared_transformer | concat | gated_attention | null | linear | Does the full balanced model win? |
| 10 | `multiscale_joint_transformer` | shared_linear | null | joint_transformer | gated_attention | null | linear | Balanced per-level versus joint 21-token processing? |
| 11 | best preset + `overlap_policy=all` | best | best | best | best | best | best | What is gained or lost by overlapping redundancy? |

The first decision gate follows orders 0–5. If multiscale simple models do not
beat the best single-scale baseline consistently, diagnose pyramid audit and
split behavior before adding Transformer capacity. If simple multiscale improves,
continue through orders 6–11.

## Explicitly deferred

The dedicated backlog records sinusoidal 2D versus no-position comparisons,
global slide-coordinate models, hierarchical explanations, per-level projections,
separate local Transformers, nonlinear level-fusion MLPs, and MIL-specific losses.
They are not required to ship the modular multiscale baseline.
