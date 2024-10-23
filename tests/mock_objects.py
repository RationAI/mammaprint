from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader


def get_dataloader_mock(
    constant_sample: torch.Tensor,
    test_slide_fp: Path,
    size: int = 64,
    batch_size: int = 32,
):
    m_dataset = MagicMock()
    m_dataset.__len__.return_value = size
    m_dataset.__getitem__.return_value = (
        constant_sample,
        constant_sample.median(),
        {
            "slide_fp": str(test_slide_fp.absolute()),
            "slide_name": "test",
            "coord_x": 0,
            "coord_y": 0,
            "tile_size": 128,
            "sample_level": 0,
        },
    )
    return DataLoader(m_dataset, batch_size=batch_size)


def get_datamodule_mock():
    """Sets up a pytorch-lightning DataModule mock object."""
    m_datamodule = MagicMock()
    m_datamodule.test_dataloader = MagicMock(
        return_value=get_dataloader_mock(
            constant_sample=torch.ones((3, 128, 128)),
            test_slide_fp=Path("tests/data/test_slide"),
        )
    )
    return m_datamodule
