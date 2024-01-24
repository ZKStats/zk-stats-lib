from abc import ABC, abstractmethod, abstractclassmethod

import numpy as np
import torch

# boolean: either 1.0 or 0.0
IsResultPrecise = torch.Tensor


class Operation(ABC):
    def __init__(self, result: torch.Tensor, error: float):
        self.result = result
        self.error = error

    @abstractclassmethod
    def create(cls, x: torch.Tensor, error: float) -> 'Operation':
        ...

    @abstractmethod
    def ezkl(self, x: torch.Tensor) -> IsResultPrecise:
        ...


class Mean(Operation):
    @classmethod
    def create(cls, x: torch.Tensor, error: float) -> 'Mean':
        return cls(torch.mean(x), error)

    def ezkl(self, x: torch.Tensor) -> IsResultPrecise:
        size = x.size()
        return torch.abs(torch.sum(x)-size[1]*self.result)<=torch.abs(self.error*size[1]*self.result)


def to_1d(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.size()
    # Only allows 1d array or [1, len(x), 1]
    if len(x_shape) == 1:
        return x
    elif len(x_shape) == 3 and x_shape[0] == 1 and x_shape[2] == 1:
        return x.reshape(-1)
    else:
        raise Exception(f"Unsupported shape: {x_shape=}")


class Median(Operation):
    def __init__(self, x: torch.Tensor, error: float):
        # NOTE: To ensure `lower` and `upper` are a scalar, `x` must be a 1d array.
        # Otherwise, if `x` is a 3d array, `lower` and `upper` will be 2d array, which are not what
        # we want in our context. However, we tend to have x as a `[1, len(x), 1]`. In this case,
        # we need to flatten `x` to 1d array to get the correct `lower` and `upper`.
        x_1d = to_1d(x)
        super().__init__(torch.tensor(np.median(x_1d)), error)
        sorted_x = np.sort(x_1d)
        len_x = len(x_1d)
        self.lower = torch.tensor(sorted_x[int(len_x/2)-1])
        self.upper = torch.tensor(sorted_x[int(len_x/2)])

    @classmethod
    def create(cls, x: torch.Tensor, error: float) -> 'Median':
        return cls(x, error)

    def ezkl(self, X: torch.Tensor) -> IsResultPrecise:
        # since within 1%, we regard as same value
        count_less = torch.sum((X < (1-self.error)*self.result).double())
        count_equal = torch.sum((torch.abs(X-self.result)<=torch.abs(self.error*self.result)).double())
        size = X.size()[1]
        half_len = torch.floor(torch.div(size, 2))

        # not support modulo yet
        less_cons = count_less<half_len+2*(size/2 - torch.floor(size/2))
        more_cons = count_less+count_equal>half_len

        # For count_equal == 0
        lower_exist = torch.sum((torch.abs(X-self.lower)<=torch.abs(self.error*self.lower)).double())>0
        lower_cons = torch.sum((X>(1+self.error)*self.lower).double())==half_len
        upper_exist = torch.sum((torch.abs(X-self.upper)<=torch.abs(self.error*self.upper)).double())>0
        upper_cons = torch.sum((X<(1-self.error)*self.upper).double())==half_len
        bound = count_less==half_len
        # 0.02 since 2*0.01
        bound_avg = (torch.abs(self.lower+self.upper-2*self.result)<=torch.abs(2*self.error*self.result))

        median_in_cons = torch.logical_and(less_cons, more_cons)
        median_out_cons = torch.logical_and(torch.logical_and(bound, bound_avg), torch.logical_and(torch.logical_and(lower_cons, upper_cons), torch.logical_and(lower_exist, upper_exist)))

        return torch.where(count_equal==0, median_out_cons, median_in_cons)


# TODO: add the rest of the operations
