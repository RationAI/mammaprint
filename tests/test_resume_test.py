import unittest
from typing import cast

from mlflow import MlflowClient

from ml.resume_test import resolve_run_id


class ResolveRunIdTest(unittest.TestCase):
    def test_rejects_missing_run_id_and_run_name_before_querying_mlflow(self) -> None:
        with self.assertRaisesRegex(SystemExit, "run id or run name"):
            resolve_run_id(
                cast("MlflowClient", object()),
                run_id=None,
                run_name=None,
                experiment="MammaPrint",
            )


if __name__ == "__main__":
    unittest.main()
