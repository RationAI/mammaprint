"""Typed representation transitions for modular multi-scale MIL pipelines."""

from dataclasses import dataclass
from typing import Literal


type FeatureAdapterName = Literal["shared_linear"] | None
type WithinLevelName = (
    Literal["mean", "max", "gated_attention", "shared_transformer"] | None
)
type LevelFusionName = (
    Literal["flatten", "concat", "mean", "gated_sum", "joint_transformer"] | None
)
type SlidePoolName = Literal["mean", "max", "gated_attention"] | None
type FinalizerName = Literal["flatten"] | None
type RepresentationKind = Literal[
    "pyramid_tokens",
    "level_vectors",
    "pyramid_vectors",
    "token_slide",
    "level_slide",
    "slide_vector",
]


_FEATURE_ADAPTERS = (None, "shared_linear")
_WITHIN_LEVELS = (None, "mean", "max", "gated_attention", "shared_transformer")


@dataclass(frozen=True)
class Representation:
    """Semantic tensor kind and its trailing or flattened feature width."""

    kind: RepresentationKind
    width: int

    @property
    def dim(self) -> int:
        """Alias for the represented feature width."""
        return self.width


@dataclass(frozen=True)
class MultiScalePipelineSpec:
    """A stage selection with validated semantic representation transitions."""

    feature_dim: int
    hidden_dim: int
    feature_adapter: FeatureAdapterName
    within_level: WithinLevelName
    level_fusion: LevelFusionName
    slide_pool: SlidePoolName
    finalizer: FinalizerName
    feature_export: bool = False

    def __post_init__(self) -> None:
        self.transitions()

    def transitions(self) -> tuple[Representation, ...]:
        """Return the initial representation and the result of every stage."""
        current = Representation("pyramid_tokens", self.feature_dim)
        states = [current]

        self._validate_name(
            "feature_adapter", self.feature_adapter, _FEATURE_ADAPTERS, current
        )
        if self.feature_adapter == "shared_linear":
            current = Representation("pyramid_tokens", self.hidden_dim)
        states.append(current)

        self._validate_name("within_level", self.within_level, _WITHIN_LEVELS, current)
        if self.within_level is not None:
            current = Representation("level_vectors", current.width)
        states.append(current)

        if current.kind == "pyramid_tokens":
            accepted_fusions: tuple[object, ...] = (
                None,
                "flatten",
                "joint_transformer",
            )
        else:
            accepted_fusions = (None, "concat", "mean", "gated_sum")
        self._validate_name(
            "level_fusion", self.level_fusion, accepted_fusions, current
        )

        if current.kind == "pyramid_tokens":
            if self.level_fusion == "flatten":
                current = Representation("pyramid_vectors", 21 * current.width)
            elif self.level_fusion == "joint_transformer":
                current = Representation("pyramid_vectors", current.width)
        elif current.kind == "level_vectors":
            if self.level_fusion == "concat":
                current = Representation("pyramid_vectors", 3 * current.width)
            elif self.level_fusion in ("mean", "gated_sum"):
                current = Representation("pyramid_vectors", current.width)
        states.append(current)

        if current.kind == "pyramid_vectors":
            accepted_pools: tuple[object, ...] = ("mean", "max", "gated_attention")
        else:
            accepted_pools = ("mean", "max")
        if self.feature_export:
            accepted_pools = (None, *accepted_pools)
        self._validate_name("slide_pool", self.slide_pool, accepted_pools, current)

        if self.slide_pool == "gated_attention":
            current = Representation("slide_vector", current.width)
        elif self.slide_pool is not None and current.kind == "pyramid_tokens":
            current = Representation("token_slide", current.width)
        elif self.slide_pool is not None and current.kind == "level_vectors":
            current = Representation("level_slide", current.width)
        elif self.slide_pool is not None and current.kind == "pyramid_vectors":
            current = Representation("slide_vector", current.width)
        states.append(current)

        if current.kind in ("pyramid_tokens", "level_vectors", "pyramid_vectors"):
            accepted_finalizers: tuple[object, ...] = (
                (None,) if self.feature_export else ()
            )
        elif current.kind in ("token_slide", "level_slide"):
            accepted_finalizers = (
                (None, "flatten") if self.feature_export else ("flatten",)
            )
        else:
            accepted_finalizers = (None, "flatten")
        self._validate_name("finalizer", self.finalizer, accepted_finalizers, current)

        if self.finalizer == "flatten":
            if current.kind == "token_slide":
                current = Representation("slide_vector", 21 * current.width)
            elif current.kind == "level_slide":
                current = Representation("slide_vector", 3 * current.width)
        states.append(current)

        return tuple(states)

    @property
    def out_dim(self) -> int | None:
        """Return the head width when the pipeline ends in a slide vector."""
        final = self.transitions()[-1]
        return final.width if final.kind == "slide_vector" else None

    def validate_head_dim(self, head_dim: int) -> None:
        """Reject a head whose input width differs from this pipeline's output."""
        final = self.transitions()[-1]
        expected = self.out_dim
        if expected != head_dim:
            raise ValueError(
                "Rejected stage 'head' for representation "
                f"{final.kind!r} (pipeline dimension {expected!r}); got head "
                f"dimension {head_dim!r}; accepted head dimension is {expected!r}. "
                f"Current spec: {self!r}."
            )

    def _validate_name(
        self,
        stage: str,
        value: object,
        accepted: tuple[object, ...],
        representation: Representation,
    ) -> None:
        if value not in accepted:
            self._reject(stage, representation, accepted)

    def _reject(
        self,
        stage: str,
        representation: Representation,
        accepted: tuple[object, ...],
    ) -> None:
        rejected = getattr(self, stage)
        raise ValueError(
            f"Rejected stage {stage!r} value {rejected!r} for representation "
            f"{representation.kind!r}; accepted alternatives are {accepted!r}. "
            f"Current spec: {self!r}."
        )


__all__ = [
    "FeatureAdapterName",
    "FinalizerName",
    "LevelFusionName",
    "MultiScalePipelineSpec",
    "Representation",
    "RepresentationKind",
    "SlidePoolName",
    "WithinLevelName",
]
