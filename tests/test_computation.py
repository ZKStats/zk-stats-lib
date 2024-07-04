from typing import Type, Callable
import statistics
import torch

import pytest

from zkstats.computation import State, Args, computation_to_model
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
    Operation
)

from .helpers import assert_result, compute, ERROR_CIRCUIT_DEFAULT, ERROR_CIRCUIT_STRICT, ERROR_CIRCUIT_RELAXED


def nested_computation(state: State, args: Args):
    x = args['columns_0']
    y = args['columns_1']
    z = args['columns_2']
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
    slope, intercept = out_10[0][0], out_10[1][0]
    reshaped = torch.cat((
        out_0.unsqueeze(0),
        out_1.unsqueeze(0),
        out_2.unsqueeze(0),
        out_3.unsqueeze(0),
        out_4.unsqueeze(0),
        out_5.unsqueeze(0),
        out_6.unsqueeze(0),
        out_7.unsqueeze(0),
        out_8.unsqueeze(0),
        out_9.unsqueeze(0),
        slope.unsqueeze(0),
        intercept.unsqueeze(0),
    )).reshape(-1,1)
    out_10 = state.mean(reshaped)
    return out_10


@pytest.mark.parametrize(
    "error",
    [ERROR_CIRCUIT_DEFAULT],
)
def test_nested_computation(tmp_path, column_0: torch.Tensor, column_1: torch.Tensor, column_2: torch.Tensor, error, scales):
    x, y, z = column_0, column_1, column_2
    state = compute(tmp_path, [x, y, z], nested_computation, scales)
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
    assert op_10.result.shape == ( 2, 1)
    assert_result(op_10.result[0][0], out_10.slope)
    assert_result(op_10.result[1][0], out_10.intercept)

    op_11 = ops[11]
    assert isinstance(op_11, Mean)
    out_11 = statistics.mean([out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10.slope, out_10.intercept])
    assert_result(torch.tensor(out_11), op_11.result)


@pytest.mark.parametrize(
    "op_type, expected_func, error",
    [
        (State.mean, statistics.mean, ERROR_CIRCUIT_DEFAULT),
        (State.median, statistics.median, ERROR_CIRCUIT_DEFAULT),
        (State.geometric_mean, statistics.geometric_mean, ERROR_CIRCUIT_DEFAULT),
        # Be more tolerant for HarmonicMean
        (State.harmonic_mean, statistics.harmonic_mean, ERROR_CIRCUIT_RELAXED),
        # Be less tolerant for Mode
        (State.mode, statistics.mode, ERROR_CIRCUIT_STRICT),
        (State.pstdev, statistics.pstdev, ERROR_CIRCUIT_DEFAULT),
        (State.pvariance, statistics.pvariance, ERROR_CIRCUIT_DEFAULT),
        (State.stdev, statistics.stdev, ERROR_CIRCUIT_DEFAULT),
        (State.variance, statistics.variance, ERROR_CIRCUIT_DEFAULT),
    ]
)
def test_computation_with_where_1d(tmp_path, error, column_0, op_type: Callable[[Operation, torch.Tensor], torch.Tensor], expected_func: Callable[[list[float]], float], scales):
    column = column_0
    def condition(_x: torch.Tensor):
        return _x < 4

    def where_and_op(state: State, args: Args):
        x = args['columns_0']
        return op_type(state, state.where(condition(x), x))
    state = compute(tmp_path, [column], where_and_op, scales)

    res_op = state.ops[-1]
    filtered = column[condition(column)]
    expected_res = expected_func(filtered.tolist())
    assert_result(res_op.result.data, expected_res)


@pytest.mark.parametrize(
    "op_type, expected_func, error",
    [
        (State.covariance, statistics.covariance, ERROR_CIRCUIT_RELAXED),
        (State.correlation, statistics.correlation, ERROR_CIRCUIT_RELAXED),
    ]
)
def test_computation_with_where_2d(tmp_path, error, column_0, column_1, op_type: Callable[[Operation, torch.Tensor], torch.Tensor], expected_func: Callable[[list[float]], float], scales):
    def condition_0(_x: torch.Tensor):
        return _x > 4

    def where_and_op(state: State, args: Args):
        x = args['columns_0']
        y = args['columns_1']
        condition_x = condition_0(x)
        filtered_x = state.where(condition_x, x)
        filtered_y = state.where(condition_x, y)
        return op_type(state, filtered_x, filtered_y)
    state = compute(tmp_path, [column_0, column_1], where_and_op, scales)

    res_op = state.ops[-1]
    condition_x = condition_0(column_0)
    filtered_x = column_0[condition_x]
    filtered_y = column_1[condition_x]
    expected_res = expected_func(filtered_x.tolist(), filtered_y.tolist())
    assert_result(res_op.result.data, expected_res)
