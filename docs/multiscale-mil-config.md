# Modular multi-scale MIL configuration

This reference configures `ml.models.aggregators.ModularMultiScaleMIL` for a
complete L5/L4/L3 embedding pyramid. The model receives a single slide bag with
shape `(R, 21, D)`: `R` retained level-5 anchor regions, and `21 = 1 + 4 + 16`
ordered tokens for L5, L4, and L3. The current pyramid data card fixes
`level_ratios: [1, 2, 4]`, `anchor_level: 5`, `level_order: [5, 4, 3]`, and
`tile_extent: 224`. `R` can differ between slides; all other axes are fixed. Before downloading
artifacts, each split derives its ordered ratios from the level cards and
checks them against `ml.aggregator.level_ratios`. Overrides to level order,
resolutions, or model ratios must agree; equal token counts alone are insufficient.

The base card is `configs/ml/aggregator/multiscale_modular.yaml`. It makes all
five stages visible, with `null` meaning identity/preserve rather than an
implicit pool or flatten:

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

`feature_dim` is the pretrained tile-embedding width (`D`, 2560 for Virchow2).
`hidden_dim` is the learned adapter/Transformer width (`H`, 512 in these
presets). `attention_dim` is the hidden width used by gated attention. The
Transformer has one layer, eight heads, MLP ratio 4.0, and dropout 0.1. The
runtime module performs the authoritative `aggregator.out_dim == head.in_dim`
check before training; `slide_dim` in a preset is a readable, independently
configured declaration of that same contract.

Training logs `trainable_parameters` directly from the instantiated model after
all configuration overrides, counting only parameters with `requires_grad=True`.
Experiment cards do not declare a fixed count, so changes to attention width,
head capacity, or constructor-time freezing are reflected in the logged value.

## Runtime controls outside the five stages

`projection_dropout` defaults to `0.1`. It is applied after the shared
`D -> H` projection and GELU only when `feature_adapter: shared_linear`; it has
no runtime effect when the adapter is `null`. As standard PyTorch dropout, it is
stochastic during training and disabled by `eval()` for validation, testing, and
prediction. Set it to `0.0` for a no-dropout projection ablation.

`feature_export` defaults to `false`, which is the setting used by every named
training preset. With this default, the pipeline must remove the variable region
axis `R`, produce one flat slide vector, and compose with a prediction head. With
`feature_export: true`, the pipeline is export-only: it may set
`slide_pool: null` to retain region features, cannot be composed with a head via
`out_dim`, rejects ordinary `forward()`, and must be called through
`export_features()`. Export mode remains explicit even when its selected stages
happen to produce a flat vector.

## Stage values, shapes, and parameters

Each arrow below describes the representation after the stage. `D'` means `D`
with no adapter and `H` with `shared_linear`.

| Stage | Allowed value | Input -> output | Parameters and use |
|---|---|---|---|
| `feature_adapter` | `null` | `(R,21,D) -> (R,21,D)` | Identity; parameter-light raw-embedding baselines. |
| | `shared_linear` | `(R,21,D) -> (R,21,H)` | Per-level LayerNorm plus one shared `D -> H` linear/GELU/dropout projection; economical common embedding space. |
| `within_level` | `null` | `(R,21,D') -> (R,21,D')` | Keeps fixed L5/L4/L3 slots for raw flattening or a joint Transformer. |
| | `mean` | `(R,21,D') -> (R,3,D')` | Parameter-free equal pooling in each 1/4/16-token group. |
| | `max` | `(R,21,D') -> (R,3,D')` | Parameter-free elementwise max, useful as a sparse-evidence control. |
| | `gated_attention` | `(R,21,D') -> (R,3,D')` | Learned gated weighted sum per level; asks whether local tile selection helps. |
| | `shared_transformer` | `(R,21,D') -> (R,3,D')` | One local Transformer shared across the three level batches; captures local geometry with bounded capacity. |
| `level_fusion` | `null` | tokens or vectors unchanged | Preserve the 21-token or 3-level axis for a later mean/max and optional final flatten. |
| | `flatten` | `(R,21,D') -> (R,21D')` | Parameter-free raw-slot flattening; only accepted from tokens. |
| | `concat` | `(R,3,D') -> (R,3D')` | Parameter-free, lossless ordered L5/L4/L3 concatenation; default after level pooling. |
| | `mean` | `(R,3,D') -> (R,D')` | Parameter-free equal-level compression control. |
| | `gated_sum` | `(R,3,D') -> (R,D')` | Learned normalized scalar weights over three levels; a lossy regularized alternative. |
| | `joint_transformer` | `(R,21,D') -> (R,D')` | Transformer over all fixed tokens; direct joint-processing comparison. |
| `slide_pool` | `null` | preserves `R` | Only valid with `feature_export: true`; normal prediction must remove `R`. |
| | `mean` | pool axis 0 | Parameter-free slide-region average; works with flat, token, or level-structured representations. |
| | `max` | pool axis 0 | Parameter-free slide-region max; works with flat, token, or level-structured representations. |
| | `gated_attention` | `(R,W) -> (W,)` | Learned scalar selection among variable-count regions; only accepts flat pyramid vectors. |
| `finalizer` | `null` | preserves final representation | Valid for already-flat slide vectors, or export-only structured outputs. |
| | `flatten` | `(21,D') -> (21D',)` or `(3,D') -> (3D',)` | Parameter-free reshape after `R` has been removed; makes a Linear/MLP-compatible slide vector. |

The vectorized local stages never loop over `R`. They operate in three batched
calls:

```text
L5: (R,  1, D') -> local module -> (R, D')
L4: (R,  4, D') -> local module -> (R, D')
L3: (R, 16, D') -> local module -> (R, D')
```

For either Transformer stage, both embedding fields must be `learned`:

```yaml
within_level: shared_transformer  # or level_fusion: joint_transformer
position_embedding: learned
level_embedding: learned
```

These are fixed-slot relative-position vectors (21 slots) and level vectors
(three levels), adding `(21 + 3) * H = 12,288` parameters at `H=512`. Every
non-Transformer preset instead explicitly uses `position_embedding: null` and
`level_embedding: null`. `sinusoidal`, `none`, or any other strings are not
accepted alternatives: the sinusoidal and no-position comparisons remain in the
[backlog](superpowers/backlog/2026-08-25-modular-multiscale-mil-backlog.md).

## Compatibility and null rules

Hydra's YAML `null` is passed as Python `None`; it is not the string `"null"`.
The pipeline validator rejects type-incompatible programs before the first
training batch:

1. `within_level: null` leaves pyramid tokens and permits only
   `level_fusion: null`, `flatten`, or `joint_transformer`.
2. A non-null `within_level` produces level vectors and permits only
   `level_fusion: null`, `concat`, `mean`, or `gated_sum`.
3. `slide_pool: gated_attention` needs a flat `(R,W)` pyramid-vector input.
4. `slide_pool: mean` and `max` also accept token and level axes.
5. `slide_pool: null` is export-only; a normal LinearHead/MLPHead cannot retain
   `R`.
6. `finalizer: flatten` needs `R` removed first. It is the required normal-run
   finalizer for a structured token/level slide result.
7. A Transformer requires learned position *and* level embeddings; embeddings
   are forbidden for non-Transformer stages.

Valid raw-layout linear baseline:

```yaml
feature_adapter: null
within_level: null
level_fusion: null
slide_pool: mean
finalizer: flatten
```

This averages regions but keeps all 21 fixed slots, producing `(21D,)`.
Invalid token/concat combination (the error names `level_fusion`, the current
`pyramid_tokens` representation, and accepted alternatives):

```yaml
within_level: null
level_fusion: concat  # invalid: concat requires (R,3,D') level vectors
```

## Named presets and widths

All cards are in `configs/ml/aggregator/multiscale_presets/`. They explicitly
spell all five stages, select the existing `LinearHead` or `MLPHead`, define a
top-level `slide_dim`, and bind `ml.head.in_dim: ${slide_dim}`. The listed
numbers assume `D=2560` and `H=512`.

| Preset | Five stages: adapter / within / fusion / slide / finalizer | Head | `slide_dim` | Question |
|---|---|---:|---:|---|
| `multiscale_mean_linear` | `null / mean / concat / mean / null` | Linear | 7680 (`3D`) | Does aligned multiscale data help without learned aggregation? |
| `multiscale_level_mean_linear` | `null / mean / mean / mean / null` | Linear | 2560 (`D`) | Is equal-level averaging sufficient? |
| `multiscale_raw_flat_linear` | `null / null / null / mean / flatten` | Linear | 53760 (`21D`) | Does the fixed 21-slot layout carry signal? |
| `multiscale_mean_mlp` | `null / mean / concat / mean / null` | MLP | 7680 (`3D`) | Does nonlinear readout help after simple concat? |
| `multiscale_projected_mean` | `shared_linear / mean / concat / mean / null` | Linear | 1536 (`3H`) | Does the shared projection bottleneck generalize better? |
| `multiscale_slide_attention` | `shared_linear / mean / concat / gated_attention / null` | Linear | 1536 (`3H`) | Is learned region selection useful? |
| `multiscale_local_attention` | `shared_linear / gated_attention / concat / gated_attention / null` | Linear | 1536 (`3H`) | Is local learned pooling useful? |
| `multiscale_transformer_mean_slide` | `shared_linear / shared_transformer / concat / mean / null` | Linear | 1536 (`3H`) | Does local geometry help without slide attention? |
| `multiscale_transformer` | `shared_linear / shared_transformer / concat / gated_attention / null` | Linear | 1536 (`3H`) | Does the balanced learned model win? |
| `multiscale_joint_transformer` | `shared_linear / null / joint_transformer / gated_attention / null` | Linear | 512 (`H`) | How does joint 21-token processing compare? |

`multiscale_raw_flat_linear` deliberately has no `flatten` fusion: it preserves
the token grid through mean slide pooling and only then flattens `(21,D)` to
`(21D,)`. `multiscale_mean_mlp` deliberately keeps the same raw mean/concat
pipeline as `multiscale_mean_linear`; changing only the existing head isolates
the effect of nonlinear cross-level readout.

`concat`, rather than level gating, is the learned-model default because there
are exactly three known, ordered levels. It preserves each vector at equal width
and lets the head learn their interaction without discarding information.
`gated_sum` compresses three vectors into one and is therefore lossy; it remains
an explicit regularized comparison, not an assumed interpretation mechanism.
Gated attention is more natural at the slide stage, where it selects among the
variable number of tissue regions and can be aligned to anchor metadata.

For the rationale and typed transition design, see the
[design](superpowers/specs/2026-08-25-modular-multiscale-mil-design.md), the
[implementation plan](superpowers/plans/2026-08-25-modular-multiscale-mil.md),
and the [deferred-comparisons backlog](superpowers/backlog/2026-08-25-modular-multiscale-mil-backlog.md).
