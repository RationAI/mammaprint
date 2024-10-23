# Standard Imports
import functools
import logging
from typing import Any

import captum
import lightning
import mlflow
import numpy as np

# Third-Party Imports
import torch
import torchvision
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._utils.lrp_rules import EpsilonRule

# Local Imports
from histopipe.trainer.callbacks.dataloader_agnostic import DataloaderAgnosticCallback
from histopipe.trainer.callbacks.image_builders import ImageBuilder


logger = logging.getLogger("ExAI")

# Captum Types
TargetType = (
    None | int | tuple[int, ...] | torch.Tensor | list[tuple[int, ...]] | list[int]
)
BaselineType = (
    None | torch.Tensor | int | float | tuple[torch.Tensor | int | float, ...]
)
StridesType = None | int | tuple[int, ...] | tuple[int | tuple[int, ...], ...]


class CaptumAttrExplainer(DataloaderAgnosticCallback):
    image_builder: ImageBuilder
    partial_image_builder: functools.partial
    explainer: Any | None
    predict_mode: str
    target: TargetType
    save_dir: str

    def __init__(
        self,
        image_builder: functools.partial,
        save_dir: str,
        target: TargetType = None,
        predict_mode="max",
    ) -> None:
        super().__init__()
        self.partial_image_builder = image_builder
        self.predict_mode = predict_mode
        self.save_dir = save_dir
        self.target = target
        self.explainer = None

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        pass

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        pass

    def _init_image_builder(self, metadata: dict):
        logger.debug("Creating new Heatmap visualizer.")
        self.image_builder = self.partial_image_builder(
            metadata=metadata, save_dir=self.save_dir
        )

    def _save_image_builder(self):
        logger.debug("Saving explanation map.")
        save_path = self.image_builder.save()
        mlflow.log_artifact(local_path=save_path, artifact_path=self.save_dir)

    def _update_exai_step(self, batch, outputs):
        x, y, metadata = batch
        target = self._resolve_target(y, outputs["outputs"])
        res = self._explain(x, target)
        data = torch.sigmoid(res).mean(dim=1, keepdim=True)
        self.image_builder.update(data=data, metadata=metadata)

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self._init_image_builder(metadata)

    def on_predict_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self._init_image_builder(metadata)

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        self._save_image_builder()

    def on_predict_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        self._save_image_builder()

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self._update_exai_step(batch, outputs)

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_predict_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self._update_exai_step(batch, outputs)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        raise NotImplementedError()

    def _resolve_target(
        self, labels: torch.Tensor, outputs: torch.Tensor
    ) -> TargetType:
        match self.target:
            case "label":
                return labels
            case "predict":
                match self.predict_mode:
                    case "max":
                        return torch.argmax(outputs, dim=-1)
                    case "min":
                        return torch.argmin(outputs, dim=-1)
                    case _:
                        raise ValueError(
                            f"Invalid predict_mode argument. Should be 'min' or 'max'; found: '{self.predict_mode}'."
                        )
            case _:
                return self.target


class OcclusionExplainer(CaptumAttrExplainer):
    sliding_window_shapes: tuple | tuple[tuple]
    strides: StridesType
    baselines: BaselineType
    perturbations_per_eval: int
    show_progress: bool

    def __init__(
        self,
        image_builder: functools.partial,
        save_dir: str,
        sliding_window_shapes: tuple,
        strides: StridesType = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        predict_mode="max",
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> None:
        super().__init__(image_builder, save_dir, target, predict_mode)
        self.sliding_window_shapes = tuple(sliding_window_shapes)
        self.strides = strides
        self.baselines = baselines
        self.perturbations_per_eval = perturbations_per_eval
        self.show_progress = show_progress

    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new Occlusion Explainer.")
        self.explainer = captum.attr.Occlusion(pl_module.model)

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        if self.baselines == "random":
            baselines = torch.rand(inputs.shape).to(inputs.get_device())
        elif self.baselines == "blur":
            baselines = torchvision.transforms.functional.gaussian_blur(
                inputs, kernel_size=3, sigma=30
            )
        else:
            baselines = self.baselines

        return self.explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=self.sliding_window_shapes,
            strides=self.strides,
            baselines=baselines,
            target=target,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=self.show_progress,
        )


class GradCAMExplainer(CaptumAttrExplainer):
    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new GradCAM Explainer.")
        self.explainer = captum.attr.GuidedGradCam(
            pl_module.model, pl_module.model.features[-3]
        )

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    @torch.enable_grad()
    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        return self.explainer.attribute(inputs=inputs, target=target)


class IntegratedGradientsExplainer(CaptumAttrExplainer):
    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new Integrated Gradient Explainer.")
        self.explainer = captum.attr.IntegratedGradients(pl_module.model)

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        return self.explainer.attribute(inputs=inputs, target=target, n_steps=25)


class LIMEExplainer(
    CaptumAttrExplainer
):  # This will make way more sense once we have segamentation done
    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new LIME Explainer.")
        self.explainer = captum.attr.Lime(
            pl_module.model, interpretable_model=SkLearnLinearRegression()
        )

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        return self.explainer.attribute(
            inputs=inputs,
            target=target,
            feature_mask=self._generate_feature_mask(inputs, superpixel_size=32),
        )

    def _generate_feature_mask(
        self, tensor: torch.Tensor, superpixel_size: int
    ) -> torch.Tensor:
        height, width = tensor.shape[2:]
        n_superpixels_h = height // superpixel_size
        n_superpixels_w = width // superpixel_size
        mask = np.zeros((height, width))
        for h in range(n_superpixels_h):
            for w in range(n_superpixels_w):
                mask[
                    h * superpixel_size : (h + 1) * superpixel_size,
                    w * superpixel_size : (w + 1) * superpixel_size,
                ] = h * n_superpixels_w + w
        return torch.tensor(mask.astype(int)).cuda().unsqueeze(0)


class GradientSHAPExplainer(CaptumAttrExplainer):
    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new GradientSHAP Explainer.")
        self.explainer = captum.attr.GradientShap(pl_module.model)

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def on_predict_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self._init_explainer(pl_module)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        return self.explainer.attribute(
            inputs=inputs, target=target, baselines=torch.zeros_like(inputs)
        )


class LRPExplainer(CaptumAttrExplainer):
    model: torch.nn.Module

    def _init_explainer(self, pl_module: lightning.LightningModule):
        logger.debug("Created new LRP Explainer.")
        self.model = pl_module.model
        self.explainer = captum.attr.LRP(pl_module.model)

    def _set_propagation_rules(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            module.rule = EpsilonRule()

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        self._set_propagation_rules(self.model)
        return self.explainer.attribute(inputs=inputs, target=target)
