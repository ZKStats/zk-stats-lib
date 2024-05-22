import torch
import torch.nn as nn

from .utils import compile_and_check


def test_onnx_to_circom(tmp_path):
    data = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self, x):
            m = torch.mean(x)  # 16
            s = torch.sum(x)  # 48
            l = torch.log(x)  # 5,3,3
            return m*s+l #773, 771, 771

    compile_and_check(Model, data, tmp_path)
