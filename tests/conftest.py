import pytest
import torch


@pytest.fixture
def error() -> float:
    return 0.01


@pytest.fixture
def column_0():
    return torch.tensor([3.0, 4.5, 1.0, 2.0, 7.5, 6.4, 5.5])


@pytest.fixture
def column_1():
    return torch.tensor([2.7, 3.3, 1.1, 2.2, 3.8, 8.2, 4.4])


@pytest.fixture
def scales():
    return [2]
