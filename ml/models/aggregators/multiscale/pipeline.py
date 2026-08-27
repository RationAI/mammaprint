"""Runtime composition for validated modular multi-scale MIL pipelines."""

from collections.abc import Sequence

from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.multiscale.adapter import (
    NullFeatureAdapter,
    SharedLinearAdapter,
    level_slices,
)
from ml.models.aggregators.multiscale.fusion import (
    ConcatLevelFusion,
    FlattenLevelFusion,
    GatedSumLevelFusion,
    JointTransformerLevelFusion,
    MeanLevelFusion,
    NullLevelFusion,
)
from ml.models.aggregators.multiscale.spec import (
    FeatureAdapterName,
    FinalizerName,
    LevelFusionName,
    MultiScalePipelineSpec,
    SlidePoolName,
    WithinLevelName,
)
from ml.models.aggregators.multiscale.within_level import (
    GatedAttentionWithinLevel,
    MaxWithinLevel,
    MeanWithinLevel,
    NullWithinLevel,
    SharedTransformerWithinLevel,
)
from ml.typing import PyramidBag


class ModularMultiScaleMIL(Aggregator[PyramidBag]):
    """Compose one validated adapter, within-level, fusion, and slide pipeline."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        level_ratios: Sequence[int],
        feature_adapter: FeatureAdapterName,
        within_level: WithinLevelName,
        level_fusion: LevelFusionName,
        slide_pool: SlidePoolName,
        finalizer: FinalizerName,
        projection_dropout: float = 0.1,
        attention_dim: int = 256,
        transformer_num_layers: int = 1,
        transformer_num_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        position_embedding: str | None = None,
        level_embedding: str | None = None,
        feature_export: bool = False,
    ) -> None:
        super().__init__()
        self.spec = MultiScalePipelineSpec(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            feature_adapter=feature_adapter,
            within_level=within_level,
            level_fusion=level_fusion,
            slide_pool=slide_pool,
            finalizer=finalizer,
            feature_export=feature_export,
        )
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.level_ratios = tuple(int(ratio) for ratio in level_ratios)
        slices = level_slices(self.level_ratios)
        self.token_count = sum(
            level_slice.stop - level_slice.start for level_slice in slices
        )
        if len(self.level_ratios) != 3 or self.token_count != 21:
            raise ValueError(
                "ModularMultiScaleMIL requires three levels containing 21 pyramid "
                f"tokens; got level_ratios={self.level_ratios!r}."
            )

        self._validate_embedding_selection(position_embedding, level_embedding)
        stage_dim = hidden_dim if feature_adapter == "shared_linear" else feature_dim

        self.feature_adapter: nn.Module
        if feature_adapter is None:
            self.feature_adapter = NullFeatureAdapter()
        else:
            self.feature_adapter = SharedLinearAdapter(
                feature_dim,
                hidden_dim,
                self.level_ratios,
                projection_dropout,
            )

        self.within_level: nn.Module
        if within_level is None:
            self.within_level = NullWithinLevel()
        elif within_level == "mean":
            self.within_level = MeanWithinLevel(self.level_ratios)
        elif within_level == "max":
            self.within_level = MaxWithinLevel(self.level_ratios)
        elif within_level == "gated_attention":
            self.within_level = GatedAttentionWithinLevel(
                stage_dim, self.level_ratios, attention_dim
            )
        else:
            if transformer_num_layers != 1:
                raise ValueError(
                    "shared_transformer within-level processing requires "
                    "transformer_num_layers=1."
                )
            self.within_level = SharedTransformerWithinLevel(
                stage_dim,
                self.level_ratios,
                transformer_num_heads,
                transformer_mlp_ratio,
                transformer_dropout,
            )

        self.level_fusion: nn.Module
        if level_fusion is None:
            self.level_fusion = NullLevelFusion()
        elif level_fusion == "flatten":
            self.level_fusion = FlattenLevelFusion()
        elif level_fusion == "concat":
            self.level_fusion = ConcatLevelFusion()
        elif level_fusion == "mean":
            self.level_fusion = MeanLevelFusion()
        elif level_fusion == "gated_sum":
            self.level_fusion = GatedSumLevelFusion(stage_dim, attention_dim)
        else:
            self.level_fusion = JointTransformerLevelFusion(
                stage_dim,
                self.level_ratios,
                transformer_num_heads,
                transformer_num_layers,
                transformer_mlp_ratio,
                transformer_dropout,
            )

        fusion_width = self.spec.transitions()[3].width
        self.slide_attention = (
            AttentionMIL(fusion_width, attention_dim)
            if slide_pool == "gated_attention"
            else None
        )

    @property
    def out_dim(self) -> int:
        if self.spec.feature_export:
            raise ValueError(
                "Feature-export pipeline is explicit export-only; use "
                "export_features() instead of composing it with a head."
            )
        out_dim = self.spec.out_dim
        if out_dim is None:
            raise ValueError(
                "Pipeline does not produce a slide vector and cannot be composed "
                "with a head."
            )
        return out_dim

    def forward(self, bag: PyramidBag) -> tuple[Tensor, Tensor | None]:
        if self.spec.feature_export:
            raise RuntimeError(
                "Feature-export pipelines must be run with export_features()."
            )
        return self._run_pipeline(bag)

    def export_features(self, bag: PyramidBag) -> tuple[Tensor, Tensor | None]:
        """Run an explicitly configured export pipeline without a prediction head."""
        if not self.spec.feature_export:
            raise RuntimeError("export_features() requires feature_export=True.")
        return self._run_pipeline(bag)

    def _run_pipeline(self, bag: PyramidBag) -> tuple[Tensor, Tensor | None]:
        self._validate_bag(bag)
        representation = self.feature_adapter(bag)
        representation = self.within_level(representation)
        fused = self.level_fusion(representation)
        representation = fused[0] if isinstance(fused, tuple) else fused

        attention = None
        if self.spec.slide_pool == "mean":
            representation = representation.mean(dim=0)
        elif self.spec.slide_pool == "max":
            representation = representation.amax(dim=0)
        elif self.spec.slide_pool == "gated_attention":
            if representation.ndim != 2:
                raise ValueError(
                    "Gated slide attention expects rank 2 (R, D) pyramid vectors; "
                    f"got rank {representation.ndim}."
                )
            assert self.slide_attention is not None
            representation, attention = self.slide_attention(representation)

        if self.spec.finalizer == "flatten":
            if self.spec.slide_pool is None:
                raise ValueError(
                    "Final flatten requires the region axis R to be removed."
                )
            representation = representation.flatten()

        return representation, attention

    def _validate_bag(self, bag: Tensor) -> None:
        if bag.ndim != 3:
            raise ValueError(
                f"ModularMultiScaleMIL expects a rank 3 (R, K, D) bag; got {bag.ndim}."
            )
        regions, tokens, features = bag.shape
        if regions == 0:
            raise ValueError("ModularMultiScaleMIL requires at least one pyramid.")
        if tokens != self.token_count:
            raise ValueError(
                f"ModularMultiScaleMIL expects {self.token_count} pyramid tokens; "
                f"got {tokens}."
            )
        if features != self.feature_dim:
            raise ValueError(
                f"ModularMultiScaleMIL expects feature width {self.feature_dim}; "
                f"got {features}."
            )

    def _validate_embedding_selection(
        self, position_embedding: str | None, level_embedding: str | None
    ) -> None:
        transformer_selected = (
            self.spec.within_level == "shared_transformer"
            or self.spec.level_fusion == "joint_transformer"
        )
        selections = {
            "position_embedding": position_embedding,
            "level_embedding": level_embedding,
        }
        if transformer_selected:
            invalid = {
                name: value for name, value in selections.items() if value != "learned"
            }
            if invalid:
                raise ValueError(
                    "Transformer stages require learned position_embedding and "
                    f"level_embedding; got {invalid!r}."
                )
        else:
            invalid = {
                name: value for name, value in selections.items() if value is not None
            }
            if invalid:
                raise ValueError(
                    "Position and level embeddings are only available for Transformer "
                    f"stages; got {invalid!r}."
                )


__all__ = ["ModularMultiScaleMIL"]
