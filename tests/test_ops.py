from typing import Type, Callable
import statistics

import pytest

import torch
from zkstats.ops import Mean, Median, GeometricMean, HarmonicMean, Mode, PStdev, PVariance, Stdev, Variance, Covariance, Correlation, LinearRegression, Operation
from zkstats.computation import IModel, IsResultPrecise

from .helpers import compute


# Error tolerance between circuit and python implementation
ERROR_CIRCUIT_DEFAULT = 0.01
ERROR_CIRCUIT_STRICT = 0.0001
ERROR_CIRCUIT_RELAXED = 0.1

# Error tolerance between zkstats python implementation and python statistics module
ERROR_ZKSTATS_STATISTICS = 0.0001


@pytest.mark.parametrize(
    "op_type, expected_func, error",
    [
        (Mean, statistics.mean, ERROR_CIRCUIT_DEFAULT),
        (Median, statistics.median, ERROR_CIRCUIT_DEFAULT),
        (GeometricMean, statistics.geometric_mean, ERROR_CIRCUIT_DEFAULT),
        # Be more tolerant for HarmonicMean
        (HarmonicMean, statistics.harmonic_mean, ERROR_CIRCUIT_RELAXED),
        # Be less tolerant for Mode
        (Mode, statistics.mode, ERROR_CIRCUIT_STRICT),
        (PStdev, statistics.pstdev, ERROR_CIRCUIT_DEFAULT),
        (PVariance, statistics.pvariance, ERROR_CIRCUIT_DEFAULT),
        (Stdev, statistics.stdev, ERROR_CIRCUIT_DEFAULT),
        (Variance, statistics.variance, ERROR_CIRCUIT_DEFAULT),
    ]
)
def test_ops_1_parameter(tmp_path, column_0: torch.Tensor, error: float, op_type: Type[Operation], expected_func: Callable[[list[float]], float], scales: list[float]):
    run_test_ops(tmp_path, op_type, expected_func, error, scales, [column_0])


@pytest.mark.parametrize(
    "op_type, expected_func, error",
    [
        (Covariance, statistics.covariance, ERROR_CIRCUIT_RELAXED),
        (Correlation, statistics.correlation, ERROR_CIRCUIT_RELAXED),
    ]
)
def test_ops_2_parameters(tmp_path, column_0: torch.Tensor, column_1: torch.Tensor, error: float, op_type: Type[Operation], expected_func: Callable[[list[float]], float], scales: list[float]):
    run_test_ops(tmp_path, op_type, expected_func, error, scales, [column_0, column_1])


def run_test_ops(tmp_path, op_type: Type[Operation], expected_func: Callable[[list[float]], float], error: float, scales: list[float], columns: list[torch.Tensor]):
    op = op_type.create(columns, error)
    expected_res = expected_func(*[column.tolist() for column in columns])
    # Check expected_res and op.result are close, within ERROR_STATISTICS
    assert abs(expected_res - op.result) < ERROR_ZKSTATS_STATISTICS * expected_res, f"{expected_res=} != {op.result=}, {ERROR_ZKSTATS_STATISTICS=}"
    class Model(IModel):
        def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
            return op.ezkl(x), op.result
    compute(tmp_path, columns, Model, scales)

