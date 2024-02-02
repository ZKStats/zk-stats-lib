from typing import Type, Callable
import statistics

import pytest

import torch
from zkstats.computation import Operation, Mean, Median, IModel, IsResultPrecise

from .helpers import compute


@pytest.mark.parametrize(
    "op_type, expected_func",
    [
        (Mean, statistics.mean),
        (Median, statistics.median),
    ]
)
def test_1d(tmp_path, column_0: torch.Tensor, error: float, op_type: Type[Operation], expected_func: Callable[[list[float]], float], scales: list[float]):
    op = op_type.create(column_0, error)
    expected_res = expected_func(column_0.tolist())
    assert expected_res == op.result
    model = op_to_model(op)
    compute(tmp_path, [column_0], model, scales)


def op_to_model(op: Operation) -> Type[IModel]:
    class Model(IModel):
        def forward(self, x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
            return op.ezkl(x), op.result
    return Model
