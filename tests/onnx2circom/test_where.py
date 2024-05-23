import torch
import torch.nn as nn

from .utils import compile_and_check

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
            # Everything works till there's swap when doing this kind of stuff: x and y somehow swap!
            return torch.where(x>8, x, y-1)
            # But these below works!
            return torch.where(x>8, x, y)
            return torch.where(torch.logical_or(x<y, x>20), x, y)
            return torch.where(x>8, x, y)
            return torch.where(x>=8, x, 11)
            return torch.where(x>=8, 0, 9)

    compile_and_check(Model, (data_1, data_2), tmp_path)
