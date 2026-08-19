from pathlib import Path
from typing import cast

import albumentations as A
import hydra
import pandas as pd
import timm
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.datasets import SlideDataset


class Virchow2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # For this, you need to setup HF_TOKEN=<X> env.variable.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()
        self.embed_dim: int = cast("int", self.module.embed_dim) * 2
        # class token + mean patch tokens

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = cast("torch.Tensor", self.module(x))  # size: B x 261 x 1280

        class_token = output[:, 0]  # size: B x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560


def save_embeddings(
    slide_tiles_embeddings: torch.Tensor,
    slide_tiles_x: torch.Tensor,
    slide_tiles_y: torch.Tensor,
    embeddings_path: Path,
) -> None:
    """Save the slide embeddings to the specified path.

    Args:
        slide_tiles_embeddings: The embeddings to save.
        slide_tiles_x: The x-coordinates of the tiles.
        slide_tiles_y: The y-coordinates of the tiles.
        embeddings_path: The path to save the embeddings to.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "x": slide_tiles_x.numpy(),
            "y": slide_tiles_y.numpy(),
            "embedding": [emb.numpy() for emb in slide_tiles_embeddings],
        }
    )

    df.to_parquet(embeddings_path, index=False, engine="pyarrow")


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dest = Path(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: Virchow2 = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    dataset = SlideDataset(
        uris=[config.dataset.path],
        transforms=A.Compose(
            [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ),
    )

    # Process each slide's tiles separately so embeddings can be saved per-slide
    for tile_dataset in tqdm(dataset.generate_datasets(), desc="Slides"):
        slide_name = tile_dataset.slide_path.stem
        embeddings_path = (dest / slide_name).with_suffix(".parquet")

        if embeddings_path.exists():
            print(f"Embeddings for slide {slide_name} already exist, skipping...")
            continue

        try:
            dataloader = DataLoader(
                tile_dataset,
                batch_size=config.dataloader.batch_size,
                num_workers=config.dataloader.num_workers,
                persistent_workers=config.dataloader.persistent_workers,
            )

            slide_tiles_embeddings = torch.zeros(
                (len(tile_dataset), tile_encoder.embed_dim), dtype=torch.float32
            )
            slide_tiles_x = torch.zeros((len(tile_dataset),), dtype=torch.int32)
            slide_tiles_y = torch.zeros((len(tile_dataset),), dtype=torch.int32)

            for i, (x, metadata) in enumerate(dataloader):
                x = x.to(device)
                embeddings = cast("torch.Tensor", tile_encoder(x))

                start = i * config.dataloader.batch_size
                end = start + embeddings.size(0)

                slide_tiles_embeddings[start:end] = embeddings.to("cpu")
                slide_tiles_x[start:end] = metadata["x"].to("cpu")
                slide_tiles_y[start:end] = metadata["y"].to("cpu")

            save_embeddings(
                slide_tiles_embeddings,
                slide_tiles_x,
                slide_tiles_y,
                embeddings_path,
            )

        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as e:
            print(f"Error processing slide {slide_name}: {e}")

    logger.log_artifacts(str(dest), artifact_path="embeddings")


if __name__ == "__main__":
    main()
