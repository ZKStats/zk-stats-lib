import torch
from typing import Type

from zkstats.computation import IState, computation_to_model_mpc

from .utils import compile_and_run_mpspdz



def computation(state: IState, args: list[torch.Tensor]):
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

    state, Model = computation_to_model_mpc(computation)

    compile_and_run_mpspdz(Model, tuple([data_1]), tmp_path)
