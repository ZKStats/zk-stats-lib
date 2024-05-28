import torch
import torch.nn as nn

from .utils import compile_and_run_mpspdz

from zkstats.computation import State, computation_to_model


def computation(state: State, args: list[torch.Tensor]):
    x = args[0]
    # y = args[1]
    # z = args[2]
    return state.mean(x)


# two tensor stuffs
def test_computation(tmp_path):
    data_1 = torch.tensor(
        [32, 8, 8],
        dtype = torch.float32,
    ).reshape(1, -1, 1)

    precal_witness_path = tmp_path / "precal_witness_path.json"
    state, Model = computation_to_model(computation, precal_witness_path, True)

    # class Model(nn.Module):
    #     def forward(self, *x):
    #         return torch.mean(x[0])

    compile_and_run_mpspdz(Model, tuple([data_1]), tmp_path)
