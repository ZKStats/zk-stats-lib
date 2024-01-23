from abc import abstractmethod
from typing import Callable, Type, Optional, Union

import torch
from torch import nn

from .ops import Operation, Mean, Median, IsResultPrecise


DEFAULT_ERROR = 0.01


class State:
    """
    State is a container for intermediate results of computation.

    Stage 1 (current_op_index is None): for every call to State (mean, median, etc.), result
        is calculated and temporarily stored in the state. Call `set_ready_for_exporting_onnx` to indicate
    Stage 2: all operations are calculated and results are ready to be used. Call `set_ready_for_exporting_onnx`
        to indicate it's ready to generate settings.
    Stage 3 (current_op_index is not None): when exporting to onnx, when the operations are called, the results and
        the conditions are popped from the state and filled in the onnx graph.
    """
    def __init__(self, error: float) -> None:
        self.ops: list[Operation] = []
        self.bools: list[Callable[[], torch.Tensor]] = []
        self.error: float = error
        # Pointer to the current operation index. If None, it's in stage 1. If not None, it's in stage 3.
        self.current_op_index: Optional[int] = None

    def set_ready_for_exporting_onnx(self) -> None:
        self.current_op_index = 0

    def mean(self, X: torch.Tensor) -> tuple[IsResultPrecise, torch.Tensor]:
        return self._call_op(X, Mean)

    def median(self, X: torch.Tensor) -> tuple[IsResultPrecise, torch.Tensor]:
        return self._call_op(X, Median)

    # TODO: add the rest of the operations

    def _call_op(self, x: torch.Tensor, op_type: Type[Operation]) -> Union[torch.Tensor, tuple[IsResultPrecise, torch.Tensor]]:
        if self.current_op_index is None:
            op = op_type.create(x, self.error)
            self.ops.append(op)
            return op.result
        else:
            # Copy the current op index to a local variable since self.current_op_index will be incremented.
            current_op_index = self.current_op_index
            # Sanity check that current op index is not out of bound
            len_ops = len(self.ops)
            if current_op_index >= len(self.ops):
                raise Exception(f"current_op_index out of bound: {current_op_index=} >= {len_ops=}")

            op = self.ops[current_op_index]
            # Sanity check that the operation type matches the current op type
            if not isinstance(op, op_type):
                raise Exception(f"operation type mismatch: {op_type=} != {type(op)=}")

            # Increment the current op index
            self.current_op_index += 1

            # Push the ezkl condition, which is checked only in the last operation
            def is_precise() -> IsResultPrecise:
                return op.ezkl(x)
            self.bools.append(is_precise)

            # If this is the last operation, aggregate all `is_precise` in `self.bools`, and return (is_precise_aggregated, result)
            # else, return only result
            if current_op_index == len_ops - 1:
                # Sanity check for length of self.ops and self.bools
                len_bools = len(self.bools)
                if len_ops != len_bools:
                    raise Exception(f"length mismatch: {len_ops=} != {len_bools=}")
                is_precise_aggregated = torch.tensor(1.0)
                for i in range(len_bools):
                    res = self.bools[i]()
                    is_precise_aggregated = torch.logical_and(is_precise_aggregated, res)
                return is_precise_aggregated, op.result
            elif current_op_index > len_ops - 1:
                # Sanity check that current op index does not exceed the length of ops
                raise Exception(f"current_op_index out of bound: {current_op_index=} > {len_ops=}")
            else:
                # It's not the last operation, just return the result
                return op.result


class IModel(nn.Module):
    @abstractmethod
    def preprocess(self, x: list[torch.Tensor]) -> None:
        ...

    @abstractmethod
    def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
        ...



# An computation function. Example:
# def computation(state: State, x: list[torch.Tensor]):
#     out_0 = state.median(x[0])
#     out_1 = state.median(x[1])
#     return state.mean(torch.tensor([out_0, out_1]).reshape(1,-1,1))
TComputation = Callable[[State, list[torch.Tensor]], tuple[IsResultPrecise, torch.Tensor]]


def create_model(computation: TComputation, error: float = DEFAULT_ERROR) -> tuple[State, Type[IModel]]:
    """
    Create a torch model from a `computation` function defined by user
    """
    state = State(error)

    class Model(IModel):
        def preprocess(self, x: list[torch.Tensor]) -> None:
            computation(state, x)
            state.set_ready_for_exporting_onnx()

        def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
            return computation(state, x)

    return state, Model
