import pytest

from tests.mock_objects import get_datamodule_mock


@pytest.fixture
def f_datamodule():
    return get_datamodule_mock()
