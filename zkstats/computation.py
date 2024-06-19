from abc import abstractmethod
from typing import Callable, Type, Optional, Union

import torch
from torch import nn
import json

from .ops import (
    Operation,
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
    IsResultPrecise,
)


DEFAULT_ERROR = 0.01
MagicNumber = 99.999


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
        self.precal_witness_path: str = None
        self.precal_witness:dict = {}
        self.isProver:bool = None
        self.op_dict:dict={}

    def set_ready_for_exporting_onnx(self) -> None:
        self.current_op_index = 0

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean of the input tensor. The behavior should conform to
        [statistics.mean](https://docs.python.org/3/library/statistics.html#statistics.mean) in Python standard library.
        """
        return self._call_op([x], Mean)

    def median(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the median of the input tensor. The behavior should conform to
        [statistics.median](https://docs.python.org/3/library/statistics.html#statistics.median) in Python standard library.
        """
        return self._call_op([x], Median)

    def geometric_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the geometric mean of the input tensor. The behavior should conform to
        [statistics.geometric_mean](https://docs.python.org/3/library/statistics.html#statistics.geometric_mean) in Python standard library.
        """
        return self._call_op([x], GeometricMean)

    def harmonic_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the harmonic mean of the input tensor. The behavior should conform to
        [statistics.harmonic_mean](https://docs.python.org/3/library/statistics.html#statistics.harmonic_mean) in Python standard library.
        """
        return self._call_op([x], HarmonicMean)

    def mode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mode of the input tensor. The behavior should conform to
        [statistics.mode](https://docs.python.org/3/library/statistics.html#statistics.mode) in Python standard library.
        """
        return self._call_op([x], Mode)

    def pstdev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the population standard deviation of the input tensor. The behavior should conform to
        [statistics.pstdev](https://docs.python.org/3/library/statistics.html#statistics.pstdev) in Python standard library.
        """
        return self._call_op([x], PStdev)

    def pvariance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the population variance of the input tensor. The behavior should conform to
        [statistics.pvariance](https://docs.python.org/3/library/statistics.html#statistics.pvariance) in Python standard library.
        """
        return self._call_op([x], PVariance)

    def stdev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sample standard deviation of the input tensor. The behavior should conform to
        [statistics.stdev](https://docs.python.org/3/library/statistics.html#statistics.stdev) in Python standard library.
        """
        return self._call_op([x], Stdev)

    def variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sample variance of the input tensor. The behavior should conform to
        [statistics.variance](https://docs.python.org/3/library/statistics.html#statistics.variance) in Python standard library.
        """
        return self._call_op([x], Variance)

    def covariance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the covariance of x and y. The behavior should conform to
        [statistics.covariance](https://docs.python.org/3/library/statistics.html#statistics.covariance) in Python standard library.
        """
        return self._call_op([x, y], Covariance)

    def correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the correlation of x and y. The behavior should conform to
        [statistics.correlation](https://docs.python.org/3/library/statistics.html#statistics.correlation) in Python standard library.
        """
        return self._call_op([x, y], Correlation)

    def linear_regression(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the linear regression of x and y. The behavior should conform to
        [statistics.linear_regression](https://docs.python.org/3/library/statistics.html#statistics.linear_regression) in Python standard library.
        """
        # hence support only one x for now
        return self._call_op([x, y], Regression)

    # WHERE operation
    def where(self, _filter: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the where operation of x. The behavior should conform to `torch.where` in PyTorch.

        :param _filter: A boolean tensor serves as a filter
        :param x: A tensor to be filtered
        :return: filtered tensor
        """
        return torch.where(_filter, x, x-x+MagicNumber)

    def _call_op(self, x: list[torch.Tensor], op_type: Type[Operation]) -> Union[torch.Tensor, tuple[IsResultPrecise, torch.Tensor]]:
        if self.current_op_index is None:
            # for prover
            if self.isProver:
                # print('Prover side create')
                op = op_type.create(x, self.error)

                # Single witness aka result
                if isinstance(op,Mean) or isinstance(op,GeometricMean) or isinstance(op, HarmonicMean) or isinstance(op, Mode):
                    op_class_str =str(type(op)).split('.')[-1].split("'")[0]
                    if op_class_str not in self.op_dict:
                        self.precal_witness[op_class_str+"_0"] = [op.result.data.item()]
                        self.op_dict[op_class_str] = 1
                    else:
                        self.precal_witness[op_class_str+"_"+str(self.op_dict[op_class_str])] = [op.result.data.item()]
                        self.op_dict[op_class_str]+=1
                elif isinstance(op, Median):
                    if 'Median' not in self.op_dict:
                        self.precal_witness['Median_0'] = [op.result.data.item(), op.lower.data.item(), op.upper.data.item()]
                        self.op_dict['Median']=1
                    else:
                        self.precal_witness['Median_'+str(self.op_dict['Median'])] = [op.result.data.item(), op.lower.data.item(), op.upper.data.item()]
                        self.op_dict['Median']+=1
                # std + variance stuffs
                elif isinstance(op, PStdev) or isinstance(op, PVariance) or isinstance(op, Stdev) or isinstance(op, Variance):
                    op_class_str =str(type(op)).split('.')[-1].split("'")[0]
                    if op_class_str not in self.op_dict:
                        self.precal_witness[op_class_str+"_0"] = [op.result.data.item(), op.data_mean.data.item()]
                        self.op_dict[op_class_str] = 1
                    else:
                        self.precal_witness[op_class_str+"_"+str(self.op_dict[op_class_str])] = [op.result.data.item(), op.data_mean.data.item()]
                        self.op_dict[op_class_str]+=1
                elif isinstance(op, Covariance):
                    if 'Covariance' not in self.op_dict:
                        self.precal_witness['Covariance_0'] = [op.result.data.item(), op.x_mean.data.item(), op.y_mean.data.item()]
                        self.op_dict['Covariance']=1
                    else:
                        self.precal_witness['Covariance_'+str(self.op_dict['Covariance'])] = [op.result.data.item(), op.x_mean.data.item(), op.y_mean.data.item()]
                        self.op_dict['Covariance']+=1
                elif isinstance(op, Correlation):
                    if 'Correlation' not in self.op_dict:
                        self.precal_witness['Correlation_0'] = [op.result.data.item(), op.x_mean.data.item(), op.y_mean.data.item(), op.x_std.data.item(), op.y_std.data.item(), op.cov.data.item()]
                        self.op_dict['Correlation']=1
                    else:
                        self.precal_witness['Correlation_'+str(self.op_dict['Correlation'])] = [op.result.data.item(), op.x_mean.data.item(), op.y_mean.data.item(), op.x_std.data.item(), op.y_std.data.item(), op.cov.data.item()]
                        self.op_dict['Correlation']+=1
                elif isinstance(op, Regression):
                    result_array = []
                    for ele in op.result.data:
                        result_array.append(ele[0].item())
                    if 'Regression' not in self.op_dict:
                        self.precal_witness['Regression_0'] = [result_array]
                        self.op_dict['Regression']=1
                    else:
                        self.precal_witness['Regression_'+str(self.op_dict['Regression'])] = [result_array]
                        self.op_dict['Regression']+=1
                    # for ele in op.result.data[0]:
                    #     result_array.append(ele[0].item())
                    # if 'Regression' not in self.op_dict:
                    #     self.precal_witness['Regression_0'] = [result_array]
                    #     self.op_dict['Regression']=1
                    # else:
                    #     self.precal_witness['Regression_'+str(self.op_dict['Regression'])] = [result_array]
                    #     self.op_dict['Regression']+=1
            # for verifier
            else:
                # print('Verifier side create')
                precal_witness = json.loads(open(self.precal_witness_path, "r").read())
                op = op_type.create(x, self.error, precal_witness, self.op_dict)
                op_class_str =str(type(op)).split('.')[-1].split("'")[0]
                if op_class_str not in self.op_dict:
                    self.op_dict[op_class_str] = 1
                else:
                    self.op_dict[op_class_str]+=1
            self.ops.append(op)
            return op.result
        else:
            # Copy the current op index to a local variable since self.current_op_index will be incremented.
            current_op_index = self.current_op_index
            # Sanity check that current op index is not out of bound
            len_ops = len(self.ops)
            if current_op_index >= len_ops:
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

            if current_op_index > len_ops - 1:
                # Sanity check that current op index does not exceed the length of ops
                raise Exception(f"current_op_index out of bound: {current_op_index=} > {len_ops=}")
            if self.isProver:
                json.dump(self.precal_witness, open(self.precal_witness_path, 'w'))
            return op.result+(x[0]-x[0])[0][0]


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
TComputation = Callable[[State, list[torch.Tensor]], torch.Tensor]


def computation_to_model(computation: TComputation, precal_witness_path:str, isProver:bool ,error: float = DEFAULT_ERROR ) -> tuple[State, Type[IModel]]:
    """
    Create a torch model from a `computation` function defined by user
    :param computation: A function that takes a State and a list of torch.Tensor, and returns a torch.Tensor
    :param error: The error tolerance for the computation.
    :return: A tuple of State and Model. The Model is a torch model that can be used for exporting to onnx.
    State is a container for intermediate results of computation, which can be useful when debugging.
    """
    state = State(error)

    state.precal_witness_path= precal_witness_path
    state.isProver = isProver

    class Model(IModel):
        def preprocess(self, x: list[torch.Tensor]) -> None:
            """
            Calculate the witnesses of the computation and store them in the state.
            """
            # In the preprocess step, the operations are calculated and the results are stored in the state.
            # So we don't need to get the returned result
            computation(state, x)
            state.set_ready_for_exporting_onnx()

        def forward(self, *x: list[torch.Tensor]) -> tuple[IsResultPrecise, torch.Tensor]:
            """
            Called by torch.onnx.export.
            """
            result = computation(state, x)
            is_computation_result_accurate = state.bools[0]()
            for op_precise_check in state.bools[1:]:
                is_op_result_accurate = op_precise_check()
                is_computation_result_accurate = torch.logical_and(is_computation_result_accurate, is_op_result_accurate)
            return is_computation_result_accurate, result
    return state, Model

