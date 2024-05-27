import pytest
import torch
import torch.nn as nn

from .utils import compile_and_run_mpspdz


@pytest.mark.parametrize(
    "func, expected_res",
    [
        pytest.param(lambda x, y: torch.where(x>8, x, y-1), torch.tensor([32, 10, 8], dtype=torch.float32), id="torch.where(x>8, x, y-1)"),
        pytest.param(lambda x, y: torch.where(x>8, x, y), torch.tensor([32, 11, 9], dtype=torch.float32), id="torch.where(x>8, x, y)"),
        pytest.param(lambda x, y: torch.where(torch.logical_or(x<y, x>20), x, y), torch.tensor([32, 8, 7], dtype=torch.float32), id="torch.where(torch.logical_or(x<y, x>20), x, y)"),
        pytest.param(lambda x, y: torch.where(x>8, x, y), torch.tensor([32, 11, 9], dtype=torch.float32), id="torch.where(x>8, x, y)"),
        pytest.param(lambda x, y: torch.where(x>=8, x, 11), torch.tensor([32, 8, 11], dtype=torch.float32), id="torch.where(x>=8, x, 11)"),
        pytest.param(lambda x, y: torch.where(x>=8, 0, 9), torch.tensor([0, 0, 9], dtype=torch.float32), id="torch.where(x>=8, 0, 9)"),
    ]
)
def test_where(func, expected_res, tmp_path):
    data_1 = torch.tensor(
        [32, 8, 7],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    data_2 = torch.tensor(
        [3, 11, 9],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self,x,y):
            return func(x,y)

    res = compile_and_run_mpspdz(Model, (data_1, data_2), tmp_path)
    output_0 = res[0]
    print(f"!@# {output_0=}")
    assert torch.allclose(output_0, expected_res), f"{output_0=}"
