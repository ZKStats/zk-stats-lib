import torch
import torch.nn as nn

from .utils import compile_and_run_mpspdz, run_torch_model


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

    # Run the model directly with torch
    output_torch = run_torch_model(Model, data)
    # Compile and run the model with MP-SPDZ
    outputs_mpspdz = compile_and_run_mpspdz(Model, data, tmp_path)
    # The model only has one output tensor
    assert len(outputs_mpspdz) == 1, f"Expecting only one output tensor, but got {len(outputs_mpspdz)} tensors."
    # Compare the output tensor with the expected output. Should be close
    assert torch.allclose(outputs_mpspdz[0], output_torch, atol=1e-3), f"Output tensor is not close to the expected output tensor. {outputs_mpspdz[0]=}, {output_torch=}"
