"""Spatially-aware Transformer MIL aggregator (TransMIL + 2D positional encoding).

The plain :class:`~ml.models.aggregators.transformer.TransformerMIL` is
permutation-invariant: it treats a slide's tiles as an unordered *set* and has no
notion of where each tile sits on the slide. This variant injects that spatial
signal by adding a **2D sinusoidal positional encoding**, computed from each tile's
physical ``(x, y)`` coordinate, to the tile features before the transformer runs.
Everything downstream (CLS-token self-attention, slide-vector readout, per-tile
attention map) is identical — this module *composes* ``TransformerMIL`` rather than
reimplementing it, so the readout has a single source of truth.

Why sinusoidal (and not TransMIL's own PPEG / a learned coordinate table):

* WSI tiles are a **sparse, non-rectangular** subset of the slide (tissue/mask
  regions only), variable in count. PPEG-style schemes reshape the token sequence
  into a dense ``sqrt(N) x sqrt(N)`` grid and convolve — inventing adjacency across real
  tissue gaps and treats *sequence order* as position. That is wrong here. A
  sinusoidal PE is a pure per-tile function of that tile's own coordinate, so gaps
  and irregular shapes cost nothing and no ordering is assumed.
* It adds **zero learned parameters**, the safest choice against overfitting on the
  small (per-slide-labelled) cohorts typical of pathology.

Coordinates arrive as the **last two columns of the bag** (``(N, feature_dim + 2)``);
the coord-aware dataset (:class:`~ml.data.datasets.spatial_scale.SpatialScaleDataset`)
packs them there. Only this aggregator, paired with that dataset, ever sees the wide
bag — the flat aggregators keep receiving a plain ``(N, feature_dim)`` tensor.

Positional context for MIL over strong frozen foundation-model features (Virchow2)
is a genuinely marginal signal for bag-of-tiles tasks (subtype / continuous-index),
so treat this as an ablation to compare against the permutation-invariant baselines,
not a default. See the module docstring rationale above for the correctness traps it
deliberately avoids.
"""

import torch
from torch import Tensor

from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.transformer import TransformerMIL
from ml.typing import Bag


class SpatialTransformerMIL(Aggregator[Bag]):
    """TransMIL self-attention with an added 2D sinusoidal positional encoding.

    The incoming bag is ``(N, feature_dim + 2)``: the first ``feature_dim`` columns
    are tile features, the last two are the tile's level-0 pixel ``(x, y)`` (top-left
    corner). Coordinates are converted to per-slide-normalised grid coordinates, a
    sinusoidal PE is built from them and added to the features, and the result is fed
    to an internal :class:`~ml.models.aggregators.transformer.TransformerMIL`.

    Args:
        feature_dim: Dimensionality of the tile **features** ``D`` (not counting the
            2 coordinate columns). Must be divisible by ``num_heads`` and by 4 (the
            PE splits ``D`` into two axis-halves, each a sin/cos pair stack).
        tile_extent: Tile edge length in tiling pixels (from the level card). With
            ``mpp`` this gives the level-0 footprint ``extent = tile_extent * mpp``
            used to turn pixel coordinates into grid indices.
        mpp: Microns-per-pixel of the tiling level (from the level card).
        num_heads: Attention heads (must divide ``feature_dim``).
        num_layers: Transformer encoder layers.
        mlp_ratio: Feed-forward width as a multiple of ``feature_dim``.
        dropout: Dropout inside the transformer layers.
        pe_scale: Multiplier on the PE before it is added to the features. Kept small
            (default 0.1) so the encoding does not dominate the feature norm.
        pe_base: Sinusoidal frequency base ``θ`` (as in Vaswani et al.).
        pe_span: Span constant mapping normalised ``[0, 1]`` coordinates into a
            usable phase range (larger = finer positional resolution).
    """

    def __init__(
        self,
        feature_dim: int,
        tile_extent: int,
        mpp: float,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pe_scale: float = 0.1,
        pe_base: float = 10000.0,
        pe_span: float = 64.0,
    ) -> None:
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by 4 for the 2D "
                "sin/cos positional encoding (two axis-halves of sin/cos pairs)."
            )
        # feature_dim % num_heads is validated inside the composed TransformerMIL.
        self.feature_dim = feature_dim
        self.extent = tile_extent * mpp
        if self.extent <= 0:
            raise ValueError(
                f"tile_extent * mpp must be positive; got {tile_extent} * {mpp}."
            )
        self.pe_scale = pe_scale
        self.pe_base = pe_base
        self.pe_span = pe_span

        # Compose (don't copy) the plain transformer: it owns the CLS token, the
        # encoder stack and the attention readout, so the interpretability map and
        # slide-vector logic stay identical to the non-spatial aggregator.
        self.transformer = TransformerMIL(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    @property
    def out_dim(self) -> int:
        return self.feature_dim

    def _positional_encoding(self, coords: Tensor) -> Tensor:
        """Build a 2D sinusoidal PE ``(N, D)`` from level-0 pixel coords ``(N, 2)``.

        Pixels -> grid indices (``/ extent``) -> per-slide bbox min-max to ``[0, 1]``
        (translation-invariant, so absolute tissue position on the glass — a scanner
        / sectioning artefact — cannot leak into the encoding). Half the channels
        encode the x axis, half the y axis; within each half, sin/cos frequency pairs.
        """
        grid = coords / self.extent  # (N, 2), float grid indices (kept continuous)

        gmin = grid.min(dim=0).values  # (2,)
        gmax = grid.max(dim=0).values  # (2,)
        # clamp_min in *grid units* (1.0 = one tile): a single tile or a bag whose
        # tiles share an x- or y-line has zero span on that axis -> normalise to 0
        # (a constant coordinate), which yields a constant PE and cannot divide by 0.
        span = (gmax - gmin).clamp_min(1.0)
        norm = (grid - gmin) / span  # (N, 2) in [0, 1]

        half = self.feature_dim // 2  # channels per axis
        num_freqs = half // 2  # sin/cos pairs per axis
        freq_idx = torch.arange(num_freqs, device=coords.device, dtype=torch.float32)
        # ω_k = 1 / θ^(2k / half); phase = normalised_coord * span * ω_k
        inv_freq = 1.0 / (self.pe_base ** (2.0 * freq_idx / half))  # (num_freqs,)

        pe = coords.new_zeros(coords.shape[0], self.feature_dim)  # (N, D)
        for axis in (0, 1):
            phase = norm[:, axis : axis + 1] * self.pe_span * inv_freq  # (N, num_freqs)
            base = axis * half
            pe[:, base : base + half : 2] = torch.sin(phase)
            pe[:, base + 1 : base + half : 2] = torch.cos(phase)
        return pe

    def forward(self, bag: Tensor) -> tuple[Tensor, Tensor]:
        # bag: (N, feature_dim + 2) — features then the (x, y) coordinate columns.
        expected = self.feature_dim + 2
        if bag.shape[-1] != expected:
            raise ValueError(
                f"SpatialTransformerMIL expects a bag of width feature_dim + 2 "
                f"({expected}); got {bag.shape[-1]}. Is it paired with "
                "SpatialScaleDataset (which appends the x/y columns)?"
            )
        features = bag[:, : self.feature_dim]  # (N, D)
        coords = bag[:, self.feature_dim :]  # (N, 2)

        pe = self._positional_encoding(coords)  # (N, D)
        # The CLS token is added *inside* the composed transformer, after this — so it
        # receives no positional term (it has no coordinate), as intended.
        return self.transformer(features + self.pe_scale * pe)


__all__ = ["SpatialTransformerMIL"]
