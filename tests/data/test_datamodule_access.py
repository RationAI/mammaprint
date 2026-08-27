import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from ml.data.datamodule import DataModule


class DataModuleDatasetAccessTest(unittest.TestCase):
    def test_get_dataset_returns_the_stage_dataset_built_by_setup(self) -> None:
        dataset = object()
        datamodule = DataModule(
            test=OmegaConf.create({"_target_": "tests.fake.Dataset"})
        )

        with patch("ml.data.datamodule.instantiate", return_value=dataset):
            datamodule.setup("test")

        self.assertIs(datamodule.get_dataset("test"), dataset)


if __name__ == "__main__":
    unittest.main()
