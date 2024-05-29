import torch
import torch.nn as nn

import pytest

from .utils import compile_and_run_mpspdz, run_torch_model


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x: x + 4, id="x + 4"),
        pytest.param(lambda x: 4 + x, id="4 + x"),
        pytest.param(lambda x: x - 4, id="x - 4"),
        pytest.param(lambda x: 4 - x, id="4 - x"),
        pytest.param(lambda x: torch.mean(x) + 4, id="mean(x) + 4"),
        pytest.param(lambda x: 4 + torch.mean(x), id="4 + mean(x)"),
        pytest.param(lambda x: torch.mean(x) - 4, id="mean(x) - 4"),
        pytest.param(lambda x: 4 - torch.mean(x), id="4 - mean(x)"),
        pytest.param(lambda x: torch.mean(x) + torch.sum(x), id="mean(x) + sum(x)"),
        pytest.param(lambda x: torch.mean(x) - torch.sum(x), id="mean(x) - sum(x)"),
        pytest.param(lambda x: torch.mean(x) + x, id="mean(x) + x"),
        pytest.param(lambda x: x + torch.mean(x), id="x + mean(x)"),
        pytest.param(lambda x: torch.mean(x) - x, id="mean(x) - x"),
        pytest.param(lambda x: x - torch.mean(x), id="x - mean(x)"),
    ]
)
def test_two_inputs(func, tmp_path):
    data = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class Model(nn.Module):
        def forward(self, x):
            # x+log(x) = 32+log(32), 8+log(8), 8+log(8)
            #          =
            return func(x)

    # Run the model directly with torch
    output_torch = run_torch_model(Model, data)
    # Compile and run the model with MP-SPDZ
    outputs_mpspdz = compile_and_run_mpspdz(Model, data, tmp_path)
    # The model only has one output tensor
    assert len(outputs_mpspdz) == 1, f"Expecting only one output tensor, but got {len(outputs_mpspdz)} tensors."
    # Compare the output tensor with the expected output. Different should be within 0.001
    assert torch.allclose(outputs_mpspdz[0], output_torch, rtol=0.001), f"Output tensor is not close to the expected output tensor. {outputs_mpspdz[0]=}, {output_torch=}"

# Not use until we support scaling for floating number
# def log(x, base=None):
#     # if base is None, we use natural logarithm
#     if base is None:
#         return torch.log(x)
#     # else, convert to `base` by `log_base(x) = log_k(x) / log_k(base)`
#     return torch.log(x) / torch.log(torch.tensor(float(base)))


# @pytest.mark.parametrize(
#     "func",
#     [
#         # pytest.param(lambda x, base: x + log(x, base), id="x + log(x)"),
#         # pytest.param(lambda x, base: log(x, base) + x, id="log(x) + x"),
#         # pytest.param(lambda x, base: x - log(x, base), id="x - log(x)"),
#         # pytest.param(lambda x, base: log(x, base) - x, id="log(x) - x"),
#         # pytest.param(lambda x, base: torch.mean(x) + log(x, base), id="mean(x) + log(x)"),
#         # pytest.param(lambda x, base: log(x, base) + torch.mean(x), id="log(x) + mean(x)"),
#         # pytest.param(lambda x, base: torch.mean(x) - log(x, base), id="mean(x) - log(x)"),
#     ]
# )

# FIXME: Now our circom interprets torch.log as base 2, while torch interprets as base e, to make things coherent, we enforce 
    # func_torch to be torch.log2. We can use log base e in circom once we support scaling.
@pytest.mark.parametrize(
    "func_mpspdz, func_torch",
    [
        pytest.param(lambda x: x + torch.log(x), lambda x: x + torch.log2(x), id="x + log(x)"),
        pytest.param(lambda x: torch.log(x) + x, lambda x: torch.log2(x) + x, id="log(x) + x"),
        pytest.param(lambda x: x - torch.log(x), lambda x: x - torch.log2(x), id="x - log(x)"),
        pytest.param(lambda x: torch.log(x) - x, lambda x: torch.log2(x) - x, id="log(x) - x"),
        pytest.param(lambda x: torch.mean(x) + torch.log(x), lambda x: torch.mean(x) + torch.log2(x), id="mean(x) + log(x)"),
        pytest.param(lambda x: torch.log(x) + torch.mean(x), lambda x: torch.log2(x) + torch.mean(x), id="log(x) + mean(x)"),
        pytest.param(lambda x: torch.mean(x) - torch.log(x), lambda x: torch.mean(x) - torch.log2(x), id="mean(x) - log(x)"),
    ]
)
def test_two_inputs_with_logs(func_mpspdz,func_torch, tmp_path):
    data = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)

    class ModelMPSPDZ(nn.Module):
        def forward(self, x):
            return func_mpspdz(x)

    outputs_tensor_mpsdpz = compile_and_run_mpspdz(ModelMPSPDZ, data, tmp_path)
    # The model only has one output tensor
    assert len(outputs_tensor_mpsdpz) == 1, f"Expecting only one output tensor, but got {len(outputs_tensor_mpsdpz)} tensors."
    output_mpspdz = outputs_tensor_mpsdpz[0]

    class ModelTorch(nn.Module):
        def forward(self, x):
            return func_torch(x)

    output_torch = run_torch_model(ModelTorch, data)
    assert output_mpspdz.shape == output_torch.shape, f"Output tensor shape is not the same. {output_mpspdz.shape=}, {output_torch.shape=}"
    # Compare the output tensor with the expected output.
    # Difference should be within 20% (|output_tensor_mpsdpz-output_torch|/|output_torch| <= error_rate)
    error_rate = 0.2
    assert torch.allclose(output_mpspdz, output_torch, rtol=error_rate), f"Output tensor is not close to the expected output tensor. {output_tensor_mpsdpz=}, {output_torch=}"
