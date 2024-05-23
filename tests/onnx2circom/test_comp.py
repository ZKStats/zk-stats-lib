import torch
import torch.nn as nn

from .utils import compile_and_check

# two tensor stuffs
def test_comparison(tmp_path):
    data_1 = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    data_2 = torch.tensor(
        [3, 8, 9],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self, x, y):
            return torch.logical_or(x<=y, x<y)
            return torch.logical_and(x<=y, x<y)
            return torch.logical_not(x<=y)
            return x>y
            return x>=y
            return x<y
            return x<=y
            return x==y

    compile_and_check(Model, (data_1, data_2), tmp_path)
