import torch
import torch.nn as nn

from .utils import compile_and_run_mpspdz

def test_where(tmp_path):
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
            # Just this example that the result are as if the x and y are swapped
            return torch.where(x>8, x, y-1)
            # But these below works!
            return torch.where(x>8, x, y)
            return torch.where(torch.logical_or(x<y, x>20), x, y)
            return torch.where(x>8, x, y)
            return torch.where(x>=8, x, 11)
            return torch.where(x>=8, 0, 9)

    res = compile_and_run_mpspdz(Model, (data_1, data_2), tmp_path)
    output_0 = res[0]
    assert torch.allclose(output_0, torch.tensor([32, 10, 8], dtype=torch.float32)), f"{output_0=}"
