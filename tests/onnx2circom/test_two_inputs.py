import torch
import torch.nn as nn

import pytest

from .utils import compile_and_check, compile_and_run_mpspdz, run_torch_model


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

    compile_and_check(Model, data, tmp_path)


def log(x, base=None):
    # if base is None, we use natural logarithm
    if base is None:
        return torch.log(x)
    # else, convert to `base` by `log_base(x) = log_k(x) / log_k(base)`
    return torch.log(x) / torch.log(torch.tensor(float(base)))


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x, base: x + log(x, base), id="x + log(x)"),
        pytest.param(lambda x, base: log(x, base) + x, id="log(x) + x"),
        pytest.param(lambda x, base: x - log(x, base), id="x - log(x)"),
        pytest.param(lambda x, base: log(x, base) - x, id="log(x) - x"),
        pytest.param(lambda x, base: torch.mean(x) + log(x, base), id="mean(x) + log(x)"),
        pytest.param(lambda x, base: log(x, base) + torch.mean(x), id="log(x) + mean(x)"),
        pytest.param(lambda x, base: torch.mean(x) - log(x, base), id="mean(x) - log(x)"),
    ]
)
def test_two_inputs_with_logs(func, tmp_path):
    e = 2.7183
    data = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)
    class ModelMPSPDZ(nn.Module):
        def forward(self, x):
            # FIXME: We should remove `log` and use `torch.log` directly when
            # we support base=e.
            # Now we need to convert base to `e` since we currently use `base=2` in mp-spdz
            # to calculate ln(x) = log_2(x) / log_2(e)
            return func(x, base=e)

    output_tensor_mpsdpz = compile_and_run_mpspdz(ModelMPSPDZ, data, tmp_path)

    class ModelTorch(nn.Module):
        def forward(self, x):
            # base = None means we don't need the conversion at all since torch uses e by default
            return func(x, base=None)

    output_torch = run_torch_model(ModelTorch, data)
    assert output_tensor_mpsdpz.shape == output_torch.shape, f"Output tensor shape is not the same. {output_tensor_mpsdpz.shape=}, {output_torch.shape=}"
    # Compare the output tensor with the expected output.
    # Difference should be within 20% (|output_tensor_mpsdpz-output_torch|/|output_torch| <= error_rate)
    error_rate = 0.2
    assert torch.allclose(output_tensor_mpsdpz, output_torch, rtol=error_rate), f"Output tensor is not close to the expected output tensor. {output_tensor_mpsdpz=}, {output_torch=}"
