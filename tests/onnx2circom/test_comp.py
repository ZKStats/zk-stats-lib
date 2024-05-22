import torch
import torch.nn as nn

from .utils import compile_and_check, compile_and_check_mult

# two tensor stuffs
def test_onnx_to_circom(tmp_path):
    data_1 = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    data_2 = torch.tensor(
        [3, 3, 5],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self, x, y):
            return x+y
            m = torch.mean(x)  # 16
            s = torch.sum(x)  # 48
            l = torch.log(x)  # 5,3,3
            return m*s+l #773, 771, 771

    compile_and_check_mult(Model, (data_1,data_2), tmp_path)
