import torch
import torch.nn as nn

import pytest

from .utils import compile_and_run_mpspdz, run_torch_model


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x : x+x.size()[0], id="x+size"),
        pytest.param(lambda x : torch.sum(x)+x.size()[0], id="sum+size"),
        pytest.param(lambda x : x[0], id="array indexing"),
        pytest.param(lambda x : x+x[0][0], id="array indexing"),
    ]
)
def test_size_index(func, tmp_path):
    data = torch.tensor(
        [-22, -8, 8],
        dtype = torch.float32,
    ).reshape( -1,1)
    data2 = torch.tensor(
        [1, 13 ],
        dtype = torch.float32,
    ).reshape( -1,1)
    class Model(nn.Module):
        def forward(self, x ):
            return func(x)

    # Run the model directly with torch
    output_torch = run_torch_model(Model, tuple([data]))
    # Compile and run the model with MP-SPDZ
    outputs_mpspdz = compile_and_run_mpspdz(Model, tuple([data]), tmp_path)
    # The model only has one output tensor
    assert len(outputs_mpspdz) == 1, f"Expecting only one output tensor, but got {len(outputs_mpspdz)} tensors."
    # Compare the output tensor with the expected output. Different should be within 0.001
    assert torch.allclose(outputs_mpspdz[0], output_torch, rtol=0.001), f"Output tensor is not close to the expected output tensor. {outputs_mpspdz[0]=}, {output_torch=}"


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x,y : y+x.size()[0], id="y+ xsize"),
        pytest.param(lambda x,y : y.size()[0]+x.size()[0], id="xsize+ysize"),
        pytest.param(lambda x,y  : x[2][0]+y[1][0], id="array indexing"),
    ]
)
def test_size_index_two_inputs(func, tmp_path):
    data = torch.tensor(
        [-22, -8, 8],
        dtype = torch.float32,
    ).reshape( -1,1)
    data2 = torch.tensor(
        [1, 13 ],
        dtype = torch.float32,
    ).reshape( -1,1)
    class Model(nn.Module):
        def forward(self, x, y ):
            return func(x,y)

    # Run the model directly with torch
    output_torch = run_torch_model(Model, tuple([data, data2]))
    # Compile and run the model with MP-SPDZ
    outputs_mpspdz = compile_and_run_mpspdz(Model, tuple([data, data2]), tmp_path)
    # The model only has one output tensor
    assert len(outputs_mpspdz) == 1, f"Expecting only one output tensor, but got {len(outputs_mpspdz)} tensors."
    # Compare the output tensor with the expected output. Different should be within 0.001
    assert torch.allclose(outputs_mpspdz[0], output_torch, rtol=0.001), f"Output tensor is not close to the expected output tensor. {outputs_mpspdz[0]=}, {output_torch=}"


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x,y : torch.cat((x,y)), id="concat x and y with dim =0"),
    ]
)
def test_cat_two_inputs(func, tmp_path):
    data = torch.tensor(
        [-22, -8, 8],
        dtype = torch.float32,
    ).reshape( -1,1)
    data2 = torch.tensor(
        [1, 13 ],
        dtype = torch.float32,
    ).reshape( -1,1)
    class Model(nn.Module):
        def forward(self, x, y ):
            return func(x,y)

    # Run the model directly with torch
    output_torch = run_torch_model(Model, tuple([data, data2]))
    # Compile and run the model with MP-SPDZ
    outputs_mpspdz = compile_and_run_mpspdz(Model, tuple([data, data2]), tmp_path)
    # The model only has one output tensor
    assert len(outputs_mpspdz) == 1, f"Expecting only one output tensor, but got {len(outputs_mpspdz)} tensors."
    # Compare the output tensor with the expected output. Different should be within 0.001
    assert torch.allclose(outputs_mpspdz[0], output_torch, rtol=0.001), f"Output tensor is not close to the expected output tensor. {outputs_mpspdz[0]=}, {output_torch=}"
