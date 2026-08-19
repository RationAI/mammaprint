"""Aggregator abstract base class.

An aggregator reduces one slide's tiles to a single slide-level vector ``(D',)``.
This is the multiple-instance-learning (MIL) pooling step; swapping the aggregator
(mean, max, gated attention, transformer, multi-scale) is the primary experimental
axis for the training skeleton.

The input is either a **flat bag** ``Tensor[N, D]`` (single level) or a
**multi-scale bag** ``list[Region]`` where each region maps ``level -> Tensor[K, D]``
(tiles across pyramid levels aligned by footprint). Flat aggregators
(mean/max/attention) accept the tensor; :class:`~ml.models.aggregators.multiscale.MultiScaleMIL`
accepts the region list. The output contract is identical, so the module and head
are agnostic to which one is in use.

Concrete aggregators live in sibling modules, one implementation per file, and
are selected via the ``ml/aggregator`` Hydra config group.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Aggregator[InputT](nn.Module, ABC):
    """Reduces one slide's tiles to a single slide vector.

    Generic (unbounded) over its input type so this base stays single-level-safe
    (it references no multilevel type): flat aggregators specialise as
    ``Aggregator[Bag]`` (a tile tensor ``(N, D)``), and the optional multi-scale
    aggregator as ``Aggregator[MultiScaleBag]`` (a list of aligned regions).
    Implementations return the pooled vector and, optionally, weights (``None`` for
    parameter-free poolings such as mean/max), surfaced for interpretability.
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Dimensionality of the aggregated slide vector."""

    @abstractmethod
    def forward(self, bag: InputT) -> tuple[Tensor, Tensor | None]:
        """Aggregate one slide's tiles.

        Args:
            bag: A flat per-tile feature tensor ``(N, D)`` (``Aggregator[Bag]``) or
                a multi-scale bag of regions (``Aggregator[MultiScaleBag]``).

        Returns:
            A tuple ``(pooled, attention)`` where ``pooled`` has shape
            ``(out_dim,)`` and ``attention`` has shape ``(N,)`` / ``(num_regions,)``
            or is ``None`` if the aggregator assigns no weights.
        """


__all__ = ["Aggregator"]
