import asyncio
import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.client import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm


async def segment_epithel(
    slides: list[str],
    tissue_masks_dir: Path,
    output_dir: Path,
    max_concurrent: int,
    timeout: int,
    model_name: str,
) -> None:
    sem = asyncio.Semaphore(max_concurrent)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _bounded_process(client: AsyncClient, path: str) -> None:
        async with sem:
            try:
                slide_filename = Path(path).with_suffix(".tiff").name
                tissue_mask_path = tissue_masks_dir / slide_filename
                output_path = output_dir / slide_filename

                await client.slide.heatmap(
                    model_name,
                    slide_path=path,
                    tissue_mask_path=tissue_mask_path,
                    output_path=output_path,
                    timeout=timeout,
                )
            except Exception as e:
                print(f"Slide processing failed for {path}: {e}, error: {e!r}")

    async with AsyncClient(timeout=timeout) as client:
        tasks = [_bounded_process(client, path) for path in slides]

        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing slides"
        ):
            await f


@with_cli_args(["+preprocessing=epithelium_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df = pd.read_csv(
        mlflow.artifacts.download_artifacts(config.dataset.mlflow_uris.dataframe)
    )
    slides: list[str] = list(slides_df["slide_path"])

    with tempfile.TemporaryDirectory(
        prefix="episeg_", dir=config.project_path
    ) as tmp_dir:
        local_tissue_masks_dir = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=config.dataset.mlflow_uris.tissue_masks,
                dst_path=tmp_dir,
            )
        )
        tmp_dir_masks = Path(tmp_dir) / "epithelium_masks"
        asyncio.run(
            segment_epithel(
                slides=slides,
                tissue_masks_dir=local_tissue_masks_dir,
                output_dir=tmp_dir_masks,
                max_concurrent=config.max_concurrent,
                timeout=config.timeout,
                model_name=config.model_name,
            )
        )
        logger.log_artifacts(
            local_dir=str(tmp_dir_masks), artifact_path="epithelium_masks"
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
