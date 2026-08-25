# Modular Multiscale MIL Backlog

This document contains ideas intentionally excluded from the first modular
multiscale implementation. An item moves into implementation only after the
initial experiment sequence establishes a justified baseline and a concrete
comparison question.

## Positional encoding ablations

### Sinusoidal 2D relative positions

Replace the learned fixed-slot position table used by Transformer variants with
deterministic 2D sine/cosine features derived from each token's normalized
location inside its level grid.

- Question: does geometric inductive bias generalize better than 21 learned slots
  with approximately 2,000 slides?
- Control: same projection, Transformer, level embeddings, optimizer, and seeds.
- Entry criterion: the learned-position Transformer must outperform the simple
  non-Transformer baseline, otherwise positional-encoding refinement is premature.

### No-position Transformer

Remove relative-position embeddings while retaining learned level embeddings.

- Question: is the improvement caused by local geometry or only by content
  interactions and magnification identity?
- Required comparison: learned position versus sinusoidal 2D versus none.
- Metric: paired seed differences against the same split and training budget.

There is literature supporting positional encoding as important for attention,
but no encoding family is guaranteed to improve this dataset. These remain
controlled empirical comparisons rather than architectural assumptions.

## Global slide coordinates

Use each pyramid's retained level-5 anchor coordinate as model input after
normalizing by slide dimensions or tissue bounds. Candidate implementations are:

1. add a 2D coordinate encoding to each `PyramidVector` before slide pooling;
2. use a spatial Transformer or graph over pyramid vectors;
3. use neighborhood-aware sparse attention for large `R`.

The current implementation preserves typed ordered anchors in prediction metadata
so these models will not require rebuilding the dataset. This is deferred because
absolute/global location answers a different question from the local aligned
pyramid and needs slide-size metadata plus careful invariance choices.

Entry criterion: the best local multiscale model should consistently improve over
single-scale controls, and an error analysis should indicate that relationships
between distant slide regions may matter.

## Hierarchical explanations

Return explanations at each learned-selection stage rather than only a prediction:

```text
slide prediction
  -> importance of each pyramid/anchor on the slide
  -> importance of L5, L4, L3 within a selected pyramid
  -> importance of the 1/4/16 tiles within each level
```

For example, a pathologist could select a high-weight anchor, inspect whether the
model relied mostly on L3 detail or L5 context, and then view the most influential
fine tiles. This requires a stable typed explanation schema and careful distinction
between attention weights and causal importance. It is deferred because concat
fusion has no explicit level weight, mean/max pools do not provide attention, and
the initial goal is predictive comparison rather than an explanation product.

Entry criterion: select a final architecture first, then define explanations only
for the stages it actually uses and validate them with occlusion or perturbation.

## Higher-capacity adapters and local models

### Per-level projection

Use separate `Linear(feature_dim, hidden_dim)` modules for L5, L4, and L3 instead
of a shared projection with level-specific normalization.

- Benefit: each magnification can learn its own feature basis.
- Cost: about three times the projection parameters and weaker cross-level sharing.
- Entry criterion: shared projection demonstrably underfits or level-specific
  feature distributions remain poorly aligned after normalization.

### Separate local Transformers

Use one Transformer per level instead of one shared Transformer.

- Benefit: maximum specialization for 1-, 4-, and 16-token patterns.
- Cost: roughly three times Transformer parameters; the L5 one-token Transformer
  is especially wasteful.
- Entry criterion: shared Transformer wins over simpler pooling and shows a clear
  level-specific failure in analysis.

### Deeper or wider local Transformer

Increase beyond the default one layer and width 512.

- Entry criterion: training and validation curves indicate underfitting, not
  overfitting, across multiple seeds.
- Guardrail: compare parameter count and wall time; H100 memory alone is not a
  reason to add capacity when the dataset has about 2,000 independent slides.

## Additional fusion modules

### MLP level fusion

Concatenate the three level vectors, then apply a nonlinear projection before
slide pooling.

- Question: are cross-level nonlinearities useful before regions are combined?
- Control: concat plus existing MLPHead tests a similar capacity allocation after
  slide pooling and should be run first.

### Cross-level attention

Treat the three level vectors as a sequence and apply a small attention block.

- Concern: with exactly three fixed vectors, concat is simpler and lossless;
  attention needs evidence that dynamic interaction is worth extra machinery.

### Alternative gated level fusion

Explore vector-valued gates or residual gated sums only if scalar `gated_sum`
beats concat or mean. Do not interpret learned gates as explanations without a
separate faithfulness test.

## Advanced MIL objectives and aggregators

Evaluate CLAM-style instance clustering, DSMIL, DTFD-MIL, TransMIL variants, or
contrastive auxiliary objectives only after the core data ablations are settled.

- These are aggregator- or loss-specific, not universal pyramid features.
- Each addition needs a single-scale capacity-matched control.
- Avoid simultaneously changing local pyramid processing, slide aggregation, and
  loss, because the source of improvement would be unidentifiable.

## Incomplete and irregular pyramids

Potential later support includes learned missing-token embeddings, masks, varying
tile extents across levels, non-integer physical ratios, or irregular magnification
sets. The current contract deliberately keeps only complete, regular pyramids.

Entry criterion: selection audits must show that completeness filtering removes a
clinically or statistically important subset that cannot be fixed upstream.

## Performance optimizations

The initial lazy reader reuses existing Parquets and vectorizes model processing.
Only profile-driven optimizations should follow:

- per-worker bounded Parquet/coordinate caches;
- memory-mapped or Arrow-native gathering;
- deduplication of repeated fine embeddings across overlapping pyramids;
- bucketing slides by retained pyramid count;
- padded multi-slide batching with masks.

These must preserve deterministic token order, anchor metadata alignment, and the
complete-pyramid audit. `overlap_policy=non_overlapping` should reduce redundancy
before adding caching complexity.

## Experiment governance

- Add five-seed confirmation only for selected finalists; use three fixed seeds
  for screening.
- Keep the test split locked until model selection is complete.
- Record parameter count, retained-pyramid audit, training wall time, peak GPU
  memory, and per-seed validation metrics for every architecture.
- Promote an item from this backlog only with a hypothesis, a capacity-matched
  control, and a rule for deciding whether to keep it.
