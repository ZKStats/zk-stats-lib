import statistics
import torch
import torch

from zkstats.computation import State, create_model
from zkstats.ops import Mean, Median

from .helpers import compute


def computation(state: State, x: list[torch.Tensor]):
    out_0 = state.median(x[0])
    out_1 = state.median(x[1])
    return state.mean(torch.tensor([out_0, out_1]).reshape(1,-1,1))


def test_computation(tmp_path, column_0: torch.Tensor, column_1: torch.Tensor, error: float):
    state, model = create_model(computation, error)
    compute(tmp_path, [column_0, column_1], model)
    assert state.current_op_index == 3

    ops = state.ops
    op0 = ops[0]
    assert isinstance(op0, Median)
    assert op0.result == statistics.median(column_0)
    op1 = ops[1]
    assert isinstance(op1, Median)
    assert op1.result == statistics.median(column_1)
    op2 = ops[2]
    assert isinstance(op2, Mean)
    assert op2.result == statistics.mean([op0.result.tolist(), op1.result.tolist()])
