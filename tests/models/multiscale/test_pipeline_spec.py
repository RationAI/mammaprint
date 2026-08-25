import unittest

from ml.models.aggregators import MultiScalePipelineSpec, Representation


class MultiScalePipelineSpecTest(unittest.TestCase):
    def test_calculates_final_dimensions_for_valid_pipelines(self) -> None:
        cases = (
            (
                MultiScalePipelineSpec(
                    feature_dim=5,
                    hidden_dim=7,
                    feature_adapter=None,
                    within_level=None,
                    level_fusion=None,
                    slide_pool="mean",
                    finalizer="flatten",
                ),
                105,
            ),
            (
                MultiScalePipelineSpec(
                    feature_dim=5,
                    hidden_dim=7,
                    feature_adapter=None,
                    within_level="mean",
                    level_fusion="concat",
                    slide_pool="mean",
                    finalizer=None,
                ),
                15,
            ),
            (
                MultiScalePipelineSpec(
                    feature_dim=5,
                    hidden_dim=7,
                    feature_adapter="shared_linear",
                    within_level="mean",
                    level_fusion="mean",
                    slide_pool="mean",
                    finalizer=None,
                ),
                7,
            ),
            (
                MultiScalePipelineSpec(
                    feature_dim=5,
                    hidden_dim=7,
                    feature_adapter="shared_linear",
                    within_level="shared_transformer",
                    level_fusion="concat",
                    slide_pool="gated_attention",
                    finalizer=None,
                ),
                21,
            ),
            (
                MultiScalePipelineSpec(
                    feature_dim=5,
                    hidden_dim=7,
                    feature_adapter="shared_linear",
                    within_level=None,
                    level_fusion="joint_transformer",
                    slide_pool="gated_attention",
                    finalizer=None,
                ),
                7,
            ),
        )

        for spec, expected in cases:
            with self.subTest(spec=spec):
                self.assertEqual(spec.out_dim, expected)
                self.assertEqual(spec.transitions()[-1].kind, "slide_vector")

    def test_transitions_include_initial_and_every_stage_state(self) -> None:
        spec = MultiScalePipelineSpec(
            feature_dim=5,
            hidden_dim=7,
            feature_adapter="shared_linear",
            within_level="mean",
            level_fusion="concat",
            slide_pool="mean",
            finalizer=None,
        )

        self.assertEqual(
            spec.transitions(),
            (
                Representation("pyramid_tokens", 5),
                Representation("pyramid_tokens", 7),
                Representation("level_vectors", 7),
                Representation("pyramid_vectors", 21),
                Representation("slide_vector", 21),
                Representation("slide_vector", 21),
            ),
        )

    def test_rejects_level_vector_fusions_for_token_input(self) -> None:
        for fusion in ("concat", "mean", "gated_sum"):
            with self.subTest(fusion=fusion):
                self._assert_invalid(
                    level_fusion=fusion,
                    stage="level_fusion",
                    representation="pyramid_tokens",
                    accepted=("None", "flatten", "joint_transformer"),
                )

    def test_rejects_token_fusions_for_level_vector_input(self) -> None:
        for fusion in ("flatten", "joint_transformer"):
            with self.subTest(fusion=fusion):
                self._assert_invalid(
                    within_level="mean",
                    level_fusion=fusion,
                    stage="level_fusion",
                    representation="level_vectors",
                    accepted=("None", "concat", "mean", "gated_sum"),
                )

    def test_rejects_gated_slide_attention_for_structured_input(self) -> None:
        self._assert_invalid(
            within_level="mean",
            level_fusion=None,
            slide_pool="gated_attention",
            stage="slide_pool",
            representation="level_vectors",
            accepted=("mean", "max"),
        )

    def test_unknown_token_fusion_lists_only_token_alternatives(self) -> None:
        self._assert_unknown_stage_alternatives(
            stage="level_fusion",
            value="typo",
            accepted=("None", "flatten", "joint_transformer"),
            excluded=("concat", "mean", "gated_sum"),
        )

    def test_unknown_structured_slide_pool_lists_only_training_alternatives(
        self,
    ) -> None:
        self._assert_unknown_stage_alternatives(
            stage="slide_pool",
            value="typo",
            accepted=("mean", "max"),
            excluded=("None", "gated_attention"),
            within_level="mean",
        )

    def test_unknown_export_finalizer_lists_only_region_preserving_alternative(
        self,
    ) -> None:
        self._assert_unknown_stage_alternatives(
            stage="finalizer",
            value="typo",
            accepted=("None",),
            excluded=("flatten",),
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            feature_export=True,
        )

    def test_rejects_null_slide_pool_for_ordinary_training(self) -> None:
        self._assert_invalid(
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            stage="slide_pool",
            representation="pyramid_vectors",
            accepted=("mean", "max", "gated_attention"),
        )

    def test_feature_export_may_retain_regions_and_has_no_head_dimension(self) -> None:
        spec = MultiScalePipelineSpec(
            feature_dim=5,
            hidden_dim=7,
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            finalizer=None,
            feature_export=True,
        )

        self.assertEqual(spec.transitions()[-1], Representation("pyramid_vectors", 15))
        self.assertIsNone(spec.out_dim)

    def test_rejects_flatten_before_removing_regions(self) -> None:
        self._assert_invalid(
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            finalizer="flatten",
            feature_export=True,
            stage="finalizer",
            representation="pyramid_vectors",
            accepted=("None",),
        )

    def test_rejects_structured_final_output_for_ordinary_training(self) -> None:
        cases = (
            (None, None, "token_slide"),
            ("mean", None, "level_slide"),
        )
        for within_level, level_fusion, representation in cases:
            with self.subTest(representation=representation):
                self._assert_invalid(
                    within_level=within_level,
                    level_fusion=level_fusion,
                    slide_pool="mean",
                    finalizer=None,
                    stage="finalizer",
                    representation=representation,
                    accepted=("flatten",),
                )

    def test_rejects_head_width_mismatch_with_full_context(self) -> None:
        spec = MultiScalePipelineSpec(
            feature_dim=5,
            hidden_dim=7,
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool="mean",
            finalizer=None,
        )

        with self.assertRaises(ValueError) as caught:
            spec.validate_head_dim(16)

        message = str(caught.exception)
        for detail in (
            "head",
            "slide_vector",
            "16",
            "15",
            "accepted",
            repr(spec),
        ):
            self.assertIn(detail, message)

    def test_none_is_the_only_null_stage_value(self) -> None:
        with self.assertRaises(ValueError) as caught:
            MultiScalePipelineSpec(
                feature_dim=5,
                hidden_dim=7,
                feature_adapter="null",  # type: ignore[arg-type]
                within_level="mean",
                level_fusion="concat",
                slide_pool="mean",
                finalizer=None,
            )

        message = str(caught.exception)
        for detail in ("feature_adapter", "pyramid_tokens", "shared_linear", "None"):
            self.assertIn(detail, message)

    def _assert_invalid(
        self,
        *,
        stage: str,
        representation: str,
        accepted: tuple[str, ...],
        within_level: str | None = None,
        level_fusion: str | None = None,
        slide_pool: str | None = "mean",
        finalizer: str | None = None,
        feature_export: bool = False,
    ) -> None:
        arguments = {
            "feature_dim": 5,
            "hidden_dim": 7,
            "feature_adapter": None,
            "within_level": within_level,
            "level_fusion": level_fusion,
            "slide_pool": slide_pool,
            "finalizer": finalizer,
            "feature_export": feature_export,
        }
        with self.assertRaises(ValueError) as caught:
            MultiScalePipelineSpec(**arguments)  # type: ignore[arg-type]

        message = str(caught.exception)
        for detail in (stage, representation, "accepted", "MultiScalePipelineSpec("):
            self.assertIn(detail, message)
        for field, value in arguments.items():
            self.assertIn(f"{field}={value!r}", message)
        for alternative in accepted:
            self.assertIn(alternative, message)

    def _assert_unknown_stage_alternatives(
        self,
        *,
        stage: str,
        value: str,
        accepted: tuple[str, ...],
        excluded: tuple[str, ...],
        within_level: str | None = None,
        level_fusion: str | None = None,
        slide_pool: str | None = "mean",
        feature_export: bool = False,
    ) -> None:
        arguments = {
            "feature_dim": 5,
            "hidden_dim": 7,
            "feature_adapter": None,
            "within_level": within_level,
            "level_fusion": level_fusion,
            "slide_pool": slide_pool,
            "finalizer": None,
            "feature_export": feature_export,
        }
        arguments[stage] = value

        with self.assertRaises(ValueError) as caught:
            MultiScalePipelineSpec(**arguments)  # type: ignore[arg-type]

        alternatives = str(caught.exception).split("Current spec:", maxsplit=1)[0]
        for alternative in accepted:
            self.assertIn(alternative, alternatives)
        for invalid_alternative in excluded:
            self.assertNotIn(invalid_alternative, alternatives)


if __name__ == "__main__":
    unittest.main()
