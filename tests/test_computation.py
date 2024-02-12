import statistics
import torch

import pytest

from zkstats.computation import State, computation_to_model
from zkstats.ops import (
    Mean,
    Median,
    GeometricMean,
    HarmonicMean,
    Mode,
    PStdev,
    PVariance,
    Stdev,
    Variance,
    Covariance,
    Correlation,
    Regression,
)

from .helpers import assert_result, compute


def nested_computation(state: State, args: list[torch.Tensor]):
    x = args[0]
    y = args[1]
    z = args[2]
    out_0 = state.median(x)
    out_1 = state.geometric_mean(y)
    out_2 = state.harmonic_mean(x)
    out_3 = state.mode(x)
    out_4 = state.pstdev(y)
    out_5 = state.pvariance(z)
    out_6 = state.stdev(x)
    out_7 = state.variance(y)
    out_8 = state.covariance(x, y)
    out_9 = state.correlation(y, z)
    out_10 = state.linear_regression(x, y)
    slope, intercept = out_10[0][0][0], out_10[0][1][0]
    reshaped = torch.tensor([
        out_0,
        out_1,
        out_2,
        out_3,
        out_4,
        out_5,
        out_6,
        out_7,
        out_8,
        out_9,
        slope,
        intercept,
    ]).reshape(1,-1,1)
    out_10 = state.mean(reshaped)
    return out_10


@pytest.mark.parametrize(
    "error",
    # TODO: if the error is larger like 0.1, we get
    # "RuntimeError: Failed to run verify: The constraint system is not satisfied"
    # Should investigate why
    [0.05],
)
def test_nested_computation(tmp_path, column_0: torch.Tensor, column_1: torch.Tensor, column_2: torch.Tensor, error, scales):
    state, model = computation_to_model(nested_computation, error)
    x, y, z = column_0, column_1, column_2
    compute(tmp_path, [x, y, z], model, scales)
    # There are 11 ops in the computation
    assert state.current_op_index == 12

    ops = state.ops
    op_0 = ops[0]
    assert isinstance(op_0, Median)
    out_0 = statistics.median(x.tolist())
    assert_result(torch.tensor(out_0), op_0.result)

    op_1 = ops[1]
    assert isinstance(op_1, GeometricMean)
    out_1 = statistics.geometric_mean(y.tolist())
    assert_result(torch.tensor(out_1), op_1.result)

    op_2 = ops[2]
    assert isinstance(op_2, HarmonicMean)
    out_2 = statistics.harmonic_mean(x.tolist())
    assert_result(torch.tensor(out_2), op_2.result)

    op_3 = ops[3]
    assert isinstance(op_3, Mode)
    out_3 = statistics.mode(x.tolist())
    assert_result(torch.tensor(out_3), op_3.result)

    op_4 = ops[4]
    assert isinstance(op_4, PStdev)
    out_4 = statistics.pstdev(y.tolist())
    assert_result(torch.tensor(out_4), op_4.result)

    op_5 = ops[5]
    assert isinstance(op_5, PVariance)
    out_5 = statistics.pvariance(z.tolist())
    assert_result(torch.tensor(out_5), op_5.result)

    op_6 = ops[6]
    assert isinstance(op_6, Stdev)
    out_6 = statistics.stdev(x.tolist())
    assert_result(torch.tensor(out_6), op_6.result)

    op_7 = ops[7]
    assert isinstance(op_7, Variance)
    out_7 = statistics.variance(y.tolist())
    assert_result(torch.tensor(out_7), op_7.result)

    op_8 = ops[8]
    assert isinstance(op_8, Covariance)
    out_8 = statistics.covariance(x.tolist(), y.tolist())
    assert_result(torch.tensor(out_8), op_8.result)

    op_9 = ops[9]
    assert isinstance(op_9, Correlation)
    out_9 = statistics.correlation(y.tolist(), z.tolist())
    assert_result(torch.tensor(out_9), op_9.result)

    op_10 = ops[10]
    assert isinstance(op_10, Regression)
    out_10 = statistics.linear_regression(x.tolist(), y.tolist())
    assert op_10.result.shape == (1, 2, 1)
    assert_result(op_10.result[0][0][0], out_10.slope)
    assert_result(op_10.result[0][1][0], out_10.intercept)

    op_11 = ops[11]
    assert isinstance(op_11, Mean)
    out_11 = statistics.mean([out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10.slope, out_10.intercept])
    assert_result(torch.tensor(out_11), op_11.result)
