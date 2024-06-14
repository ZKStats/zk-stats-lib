import torch
from typing import Type

from zkstats.computation import IState, computation_to_model_mpc, MagicNumber

from .utils import compile_and_run_mpspdz


def computation(state: IState, args: list[torch.Tensor]):
    columns_0 = [args[0], args[1]]
    columns_1 = [args[2], args[3]]
    x_key = columns_0[0]
    y_key = columns_1[0]
    num_rows_x = x_key.size(0)
    num_cols_y = len(columns_1)

    # Create a tensor for each new y columns
    new_y = [torch.where(x_key == 0, 0, 0) for _ in columns_1]
    for i in range(num_rows_x):
        # i = 0, one_hot = [1, 0, 0, 0]
        # i = 1, one_hot = [0, 1, 0, 0]
        one_hot = torch.arange(num_rows_x) == i
        # new_y[i] = 1
        # [1, 4] -> [0, 1]
        mask = y_key == x_key[i]
        # is_mask_nonzero = torch.sum(mask) != 0
        for k in range(num_cols_y):
            # [0, 1] * [5, 4] -> [0, 4]
            # sum([0, 4]) -> 4
            matched_value = torch.sum(mask * columns_1[k])
            # [1, 0, 0, 0] * 4 -> [4, 0, 0, 0]
            entry = one_hot * matched_value
            new_y[k] = entry + new_y[k]
            # new_y[k][i] = matched_value + new_y[k][i]
    return list(columns_0) + [torch.where(col == 0, MagicNumber, col) for col in new_y]


# two tensor stuffs
def test_computation(tmp_path):
    x_0 = torch.tensor(
        [1, 2, 3],
        dtype = torch.float32,
    ).reshape(-1, 1)
    x_1 = torch.tensor(
        [180, 160, 183],
        dtype = torch.float32,
    ).reshape(-1, 1)
    y_0 = torch.tensor(
        [1, 2, 4],
        dtype = torch.float32,
    ).reshape(-1, 1)
    y_1 = torch.tensor(
        [50, 40, 75],
        dtype = torch.float32,
    ).reshape(-1, 1)

    data = (x_0, x_1, y_0, y_1)

    state, Model = computation_to_model_mpc(computation)

    res = compile_and_run_mpspdz(Model, data, tmp_path)
    print(f"!@# res={res}")
