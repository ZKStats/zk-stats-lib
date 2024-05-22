import torch
import torch.nn as nn

import pytest

from .utils import compile_and_check, compile_and_check_mult


@pytest.mark.parametrize(
    "func",
    [
        # pytest.param(lambda x: x + 4, id="x + 4"),
        # pytest.param(lambda x: 4 + x, id="4 + x"),
        # pytest.param(lambda x: x - 4, id="x - 4"),
        # pytest.param(lambda x: 4 - x, id="4 - x"),
        # pytest.param(lambda x: torch.mean(x) + 4, id="mean(x) + 4"),
        # pytest.param(lambda x: 4 + torch.mean(x), id="4 + mean(x)"),
        # pytest.param(lambda x: torch.mean(x) - 4, id="mean(x) - 4"),
        # pytest.param(lambda x: 4 - torch.mean(x), id="4 - mean(x)"),
        # pytest.param(lambda x: torch.mean(x) + torch.sum(x), id="mean(x) + sum(x)"),
        # pytest.param(lambda x: torch.mean(x) - torch.sum(x), id="mean(x) - sum(x)"),
        # pytest.param(lambda x: torch.mean(x) + x, id="mean(x) + x"),
        # pytest.param(lambda x: x + torch.mean(x), id="x + mean(x)"),
        # pytest.param(lambda x: torch.mean(x) - x, id="mean(x) - x"),
        # pytest.param(lambda x: x - torch.mean(x), id="x - mean(x)"),
        # `log` is slower. We can test non-slow ones with `pytest -m "not slow"`
        pytest.param(lambda x: x + torch.log(x), id="x + log(x)"),
        # pytest.param(lambda x: torch.log(x) + x, id="log(x) + x"),
        # pytest.param(lambda x: x - torch.log(x), id="x - log(x)"),
        # pytest.param(lambda x: torch.log(x) - x, id="log(x) - x"),
        # pytest.param(lambda x: torch.mean(x) + torch.log(x), id="mean(x) + log(x)"),
        # pytest.param(lambda x: torch.log(x) + torch.mean(x), id="log(x) + mean(x)"),
        # pytest.param(lambda x: torch.mean(x) - torch.log(x), id="mean(x) - log(x)"),
    ]
)
def test_two_inputs(func, tmp_path):
    data = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self, x):
            return func(x)

    compile_and_check_mult(Model, data, tmp_path)
