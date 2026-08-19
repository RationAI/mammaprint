import asyncio
import tempfile
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


async def put_request(
    session: ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
    data: dict[str, Any],
) -> tuple[int, str]:
    """Execute a PUT request with timeout handling."""
    timeout = ClientTimeout(total=request_timeout)

    try:
        async with (
            semaphore,
            session.put(url, json=data, timeout=timeout) as response,
        ):
            result = await response.text()

            print(
                f"Processed {data['wsi_path']}:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )

            return response.status, result
    except TimeoutError:
        slide_name = Path(data["wsi_path"]).name
        print(
            f"Request to {url} timed out after {request_timeout} seconds. Slide: {slide_name}"
        )
        return -1, "Timeout"
    except Exception as exc:  # noqa: BLE001 - return a failed request result
        slide_name = Path(data.get("wsi_path", "<unknown>")).name
        print(f"Request to {url} failed for {slide_name}: {exc}")
        return -1, str(exc)


async def repeatable_put_request(
    session: ClientSession,
    url: str,
    data: dict[str, Any],
    num_repeats: int,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
) -> None:
    """Execute a PUT request with retry logic for handling failures."""
    for attempt in range(1, num_repeats + 1):
        status, text = await put_request(session, url, semaphore, request_timeout, data)

        if status == -1 and text == "Timeout":
            return

        if status == 500 and "Internal Server Error" in text:
            att_count = f"attempt {attempt}/{num_repeats}"
            print(
                f"Unexpected status 500 received for {data.get('wsi_path', '<unknown>')} ({att_count}):\n\tResponse: {text}\n"
            )
            await asyncio.sleep(2**attempt)

            continue

        print(
            f"Processed {data.get('wsi_path', '<unknown>')}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )

        return

    print(
        f"Failed to process {data.get('wsi_path', '<unknown>')}:\n\tAll retry attempts failed\n"
    )


async def generate_report(
    session: ClientSession,
    report_request_timeout: int,
    slides: list[Path],
    output_dir: str,
    save_location: str,
    url: str,
    semaphore: asyncio.Semaphore,
    compute_metrics: bool,
) -> None:
    """Generate quality control report for all processed slides."""
    url = url + "report"

    data = {
        "backgrounds": slides,
        "mask_dir": output_dir,
        "save_location": save_location,
        "compute_metrics": compute_metrics,
    }

    try:
        async with (
            semaphore,
            session.put(
                url,
                json=data,
                timeout=ClientTimeout(total=report_request_timeout),
            ) as response,
        ):
            result = await response.text()

            print(
                f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )
    except TimeoutError:
        print(
            f"Report generation request to {url} timed out after {report_request_timeout} seconds."
        )
    except Exception as exc:  # noqa: BLE001 - report generation is best-effort
        print(f"Report generation request to {url} failed: {exc}")


async def qc_main(
    config: DictConfig,
    slides: list[Path],
    report_path: str,
    logger: MLFlowLogger,
    semaphore: asyncio.Semaphore,
) -> None:
    """Main async function to orchestrate quality control checks for all slides."""
    output_path = config.output_path
    url = config.url

    async with ClientSession() as session:
        tasks = [
            repeatable_put_request(
                session=session,
                request_timeout=config.request_timeout,
                url=url,
                num_repeats=config.num_repeats,
                semaphore=semaphore,
                data={
                    "wsi_path": str(slide),
                    "output_path": output_path,
                    "mask_level": config.mask_level,
                    "sample_level": config.sample_level,
                    "check_residual": True,
                    "check_folding": True,
                    "check_blur": True,
                },
            )
            for slide in slides
        ]

        await asyncio.gather(*tasks)

        await generate_report(
            session=session,
            slides=slides,
            output_dir=output_path,
            save_location=report_path,
            url=url,
            semaphore=semaphore,
            report_request_timeout=config.report_request_timeout,
            compute_metrics=True,
        )

        try:
            logger.experiment.log_artifacts(
                run_id=logger.run_id, local_dir=Path(report_path).parent.as_posix()
            )
        except Exception:  # noqa: BLE001 - artifact upload must not abort QC
            print(f"Could not upload report artifact from {report_path}")


@with_cli_args(["+preprocessing=qc_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    """Entry point for quality control preprocessing."""
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(config.dataset.paths.data_mapping)
    slides = [Path(path + ".mrxs") for path in df["path"]]

    semaphore = asyncio.Semaphore(config.request_limit)

    with tempfile.TemporaryDirectory(
        prefix="qc_masks_report", dir=Path(config.project_path).as_posix()
    ) as tmp_dir:
        report_path = Path(tmp_dir) / "report.html"

        report_path.parent.mkdir(parents=True, exist_ok=True)

        asyncio.run(
            qc_main(
                config=config,
                slides=slides,
                report_path=report_path.absolute().as_posix(),
                logger=logger,
                semaphore=semaphore,
            )
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
