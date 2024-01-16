from abc import abstractmethod
from typing import Callable, Type

import torch
from torch import nn

from .ops import Operation, Mean, Median, IsResultPrecise


DEFAULT_ERROR = 0.01


class State:
    """
    State is a container for intermediate results of computation.

    Stage 1 (is_exporting_onnx = False): for every call to State (mean, median, etc.), result
        is calculated and temporarily stored in the state. Call `set_ready_for_exporting_onnx` to indicate
    Stage 2: all operations are calculated and results are ready to be used. Call `set_ready_for_exporting_onnx`
        to indicate it's ready to generate settings.
    Stage 3 (is_exporting_onnx = True): when exporting to onnx, when the operations are called, the results and
        the conditions are popped from the state and filled in the onnx graph.
    """
    def __init__(self, error = DEFAULT_ERROR) -> None:
        self.ops: list[Operation] = []
        self.error: float = error
        self.is_exporting_onnx = False

    def set_ready_for_exporting_onnx(self) -> None:
        self.is_exporting_onnx = True

    def mean(self, X: torch.Tensor) -> tuple[IsResultPrecise, torch.Tensor]:
        return self._call_op(X, Mean)

    def median(self, X: torch.Tensor) -> tuple[IsResultPrecise, torch.Tensor]:
        return self._call_op(X, Median)

    # TODO: add the rest of the operations

    def _call_op(self, x: torch.Tensor, op_type: Type[Operation]) -> tuple[IsResultPrecise, torch.Tensor]:
        if self.is_exporting_onnx is False:
            op = op_type.create(x, self.error)
            self.ops.append(op)
            return torch.tensor(1), op.result
        else:
            op = self.ops.pop(0)
            if not isinstance(op, op_type):
                raise Exception(f"operation type mismatch: {op_type=} != {type(op)=}")
            return op.ezkl(x), op.result


class IModel(nn.Module):
    @abstractmethod
    def preprocess(self, x: list[torch.Tensor]) -> None:
        ...

    @abstractmethod
    def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
        ...



# An computation function. Example:
# def computation(state: State, x: list[torch.Tensor]):
#     b_0, out_0 = state.median(x[0])
#     b_1, out_1 = state.median(x[1])
#     b_2, out_2 = state.mean(torch.tensor([out_0, out_1]).reshape(1,-1,1))
#     return torch.logical_and(torch.logical_and(b_0, b_1), b_2), out_2
TComputation = Callable[[State, list[torch.Tensor]], tuple[IsResultPrecise, torch.Tensor]]



def create_model(computation: TComputation) -> Type[IModel]:
    """
    Create a torch model from a `computation` function defined by user
    """
    state = State()

    class Model(IModel):
        def preprocess(self, x: list[torch.Tensor]) -> None:
            computation(state, x)
            state.set_ready_for_exporting_onnx()

        def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
            return computation(state, x)

    return Model
