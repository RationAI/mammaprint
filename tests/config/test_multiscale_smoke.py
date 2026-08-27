import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from ml.data import mil_collate
from ml.models import MammaprintModule
from ml.typing import PyramidBag, PyramidSample, PyramidSlideMetadata


CONFIGS = Path(__file__).parents[2] / "configs"


class MultiscaleSmokeFitTest(unittest.TestCase):
    def test_mean_and_transformer_presets_fit_two_slides_on_cpu(self) -> None:
        cases = (
            ("multiscale_mean_linear", False),
            ("multiscale_transformer", True),
        )
        for preset, has_attention in cases:
            with self.subTest(preset=preset):
                module = self._module(preset)
                samples = self._samples()
                loader = DataLoader(
                    samples,
                    batch_size=2,
                    shuffle=False,
                    collate_fn=mil_collate,
                )
                trainer = Trainer(
                    max_epochs=1,
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    num_sanity_val_steps=0,
                    limit_train_batches=1,
                    limit_val_batches=1,
                )

                trainer.fit(
                    module,
                    train_dataloaders=loader,
                    val_dataloaders=loader,
                )

                loss = trainer.callback_metrics["train/loss_epoch"]
                self.assertTrue(torch.isfinite(loss), loss)
                module.eval()
                with torch.no_grad():
                    output = module.predict_step(samples, batch_idx=0)
                self.assertEqual(
                    [metadata["slide_id"] for metadata in output["metadata"]],
                    ["slide-a", "slide-b"],
                )
                self.assertEqual(
                    [metadata["anchors"] for metadata in output["metadata"]],
                    [
                        ((112, 224), (336, 448)),
                        ((560, 672), (784, 896), (1008, 1120)),
                    ],
                )
                expected_lengths = (2, 3)
                for attention, expected_length in zip(
                    output["attentions"], expected_lengths, strict=True
                ):
                    if has_attention:
                        self.assertIsNotNone(attention)
                        assert attention is not None
                        self.assertEqual(tuple(attention.shape), (expected_length,))
                    else:
                        self.assertIsNone(attention)

    @staticmethod
    def _module(preset: str) -> MammaprintModule[PyramidBag, PyramidBag]:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            config = compose(
                config_name="ml",
                overrides=[
                    f"+experiment=ml/multiscale/{preset}",
                    "feature_dim=8",
                    "multiscale_hidden_dim=8",
                    "slide_dim=24",
                ],
            )
        return instantiate(config.ml)

    @staticmethod
    def _samples() -> list[PyramidSample]:
        metadata: tuple[PyramidSlideMetadata, ...] = (
            {
                "slide_id": "slide-a",
                "anchors": ((112, 224), (336, 448)),
            },
            {
                "slide_id": "slide-b",
                "anchors": ((560, 672), (784, 896), (1008, 1120)),
            },
        )
        bags = (
            PyramidBag(torch.linspace(-1.0, 1.0, 2 * 21 * 8).reshape(2, 21, 8)),
            PyramidBag(torch.linspace(1.0, -1.0, 3 * 21 * 8).reshape(3, 21, 8)),
        )
        labels = (torch.tensor([0.0]), torch.tensor([1.0]))
        return list(zip(bags, labels, metadata, strict=True))


if __name__ == "__main__":
    unittest.main()
