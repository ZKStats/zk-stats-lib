from abc import ABC, abstractmethod, abstractclassmethod
import statistics
from typing import Optional

import numpy as np
import torch

# boolean: either 1.0 or 0.0
IsResultPrecise = torch.Tensor
MagicNumber = 99.999


class Operation(ABC):
    def __init__(self, result: torch.Tensor, error: float):
        self.result = torch.nn.Parameter(data=result, requires_grad=False)
        self.error = error

    @abstractclassmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Operation':
        ...

    @abstractmethod
    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        ...



class Mean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float, precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None ) -> 'Mean':
        # support where statement, hopefully we can use 'nan' once onnx.isnan() is supported
        if precal_witness is None:
            # this is prover
            # print('provvv')
            return cls(torch.mean(x[0][x[0]!=MagicNumber]), error)
        else:
            # this is verifier
            # print('verrrr')
            if op_dict is None:
                return cls(torch.tensor(precal_witness['Mean_0'][0]), error)
            elif 'Mean' not in op_dict:
                return cls(torch.tensor(precal_witness['Mean_0'][0]), error)
            else:
                return cls(torch.tensor(precal_witness['Mean_'+str(op_dict['Mean'])][0]), error)


    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x = torch.where(x==MagicNumber, 0.0, x)
        return torch.abs(torch.sum(x)-size*self.result)<=torch.abs(self.error*self.result*size)


def to_1d(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.size()
    # Only allows 1d array or [len(x), 1]
    if len(x_shape) == 1:
        return x
    elif len(x_shape) == 2 and x_shape[1] == 1:
        return x.reshape(-1)
    else:
        raise Exception(f"Unsupported shape: {x_shape=}")


class Median(Operation):
    def __init__(self, x: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None ):
        if precal_witness is None:
        # NOTE: To ensure `lower` and `upper` are a scalar, `x` must be a 1d array.
        # Otherwise, if `x` is a 3d array, `lower` and `upper` will be 2d array, which are not what
        # we want in our context. However, we tend to have x as a `[1, len(x), 1]`. In this case,
        # we need to flatten `x` to 1d array to get the correct `lower` and `upper`.
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            super().__init__(torch.tensor(np.median(x_1d)), error)
            sorted_x = np.sort(x_1d)
            len_x = len(x_1d)
            self.lower = torch.nn.Parameter(data = torch.tensor(sorted_x[int(len_x/2)-1], dtype = torch.float32), requires_grad=False)
            self.upper = torch.nn.Parameter(data = torch.tensor(sorted_x[int(len_x/2)], dtype = torch.float32), requires_grad=False)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['Median_0'][0]), error)
                self.lower = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_0'][1]), requires_grad=False)
                self.upper = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_0'][2]), requires_grad=False)             
            elif 'Median' not in op_dict:
                super().__init__(torch.tensor(precal_witness['Median_0'][0]), error)
                self.lower = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_0'][1]), requires_grad=False)
                self.upper = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_0'][2]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['Median_'+str(op_dict['Median'])][0]), error)
                self.lower = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_'+str(op_dict['Median'])][1]), requires_grad=False)
                self.upper = torch.nn.Parameter(data = torch.tensor(precal_witness['Median_'+str(op_dict['Median'])][2]), requires_grad=False)


    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None ) -> 'Median':
        return cls(x[0],error, precal_witness, op_dict)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        old_size = x.size()[0]
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        min_x = torch.min(x)
        x = torch.where(x==MagicNumber,min_x-1, x)

        # since within 1%, we regard as same value
        count_less = torch.sum(torch.where(x < self.result, 1.0, 0.0))-(old_size-size)
        count_equal = torch.sum(torch.where(x==self.result, 1.0, 0.0))
        half_size = torch.floor(torch.div(size, 2))
        # print('hhhh: ', half_size)
        less_cons = count_less<half_size+size%2
        more_cons = count_less+count_equal>half_size

        # For count_equal == 0
        lower_exist = torch.sum(torch.where(x==self.lower, 1.0, 0.0))>0
        lower_cons = torch.sum(torch.where(x>self.lower, 1.0, 0.0))==half_size
        upper_exist = torch.sum(torch.where(x==self.upper, 1.0, 0.0))>0
        upper_cons = torch.sum(torch.where(x<self.upper, 1.0, 0.0))==half_size
        bound = count_less== half_size
        # 0.02 since 2*0.01
        bound_avg = (torch.abs(self.lower+self.upper-2*self.result)<=torch.abs(2*self.error*self.result))

        median_in_cons = torch.logical_and(less_cons, more_cons)
        median_out_cons = torch.logical_and(torch.logical_and(bound, bound_avg), torch.logical_and(torch.logical_and(lower_cons, upper_cons), torch.logical_and(lower_exist, upper_exist)))
        return torch.where(count_equal==0.0, median_out_cons, median_in_cons)


class GeometricMean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'GeometricMean':
        if precal_witness is None:
            x_1d = to_1d(x[0])
            x_1d = x_1d[x_1d!=MagicNumber]
            result = torch.exp(torch.mean(torch.log(x_1d)))
            return cls(result, error)
        else:
            if op_dict is None:
                return cls(torch.tensor(precal_witness['GeometricMean_0'][0]), error)
            elif 'GeometricMean' not in op_dict:
                return cls(torch.tensor(precal_witness['GeometricMean_0'][0]), error)
            else:
                return cls(torch.tensor(precal_witness['GeometricMean_'+str(op_dict['GeometricMean'])][0]), error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [n, 1]
        x = x[0]
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x = torch.where(x==MagicNumber, 1.0, x)
        return torch.abs((torch.log(self.result)*size)-torch.sum(torch.log(x)))<=size*torch.log(torch.tensor(1+self.error))


class HarmonicMean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float, precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'HarmonicMean':
        if precal_witness is None:
            x_1d = to_1d(x[0])
            x_1d = x_1d[x_1d!=MagicNumber]
            result = torch.div(1.0,torch.mean(torch.div(1.0, x_1d)))
            return cls(result, error)
        else:
            if op_dict is None:
                return cls(torch.tensor(precal_witness['HarmonicMean_0'][0]), error)
            elif 'HarmonicMean' not in op_dict:
                return cls(torch.tensor(precal_witness['HarmonicMean_0'][0]), error)
            else:
                return cls(torch.tensor(precal_witness['HarmonicMean_'+str(op_dict['HarmonicMean'])][0]), error)
     

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [n, 1]
        x = x[0]
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        return torch.abs((self.result*torch.sum(torch.where(x==MagicNumber, 0.0, torch.div(1.0, x)))) - size)<=torch.abs(self.error*size)


def mode_within(data_array: torch.Tensor, error: float) -> torch.Tensor:
    """
    Find the mode (the single most common data point) from the data_array.
    :param data_array: The data array.
    :param error: The error that allows the data point to be considered the same.
       For example, if error = 0.01, then 0.999 and 1.000 are considered the same.
    """
    max_sum_freq = 0
    mode = data_array[0]
    # print("arrrrr: ", data_array)
    # print("seetttt: ", torch.unique(data_array))
    for check_val in data_array:
        sum_freq = sum(1 for ele in data_array if abs(ele - check_val) <= abs(error * check_val))
        if sum_freq > max_sum_freq:
            mode = check_val
            max_sum_freq = sum_freq
    return mode


# TODO: Add class Mode_within , different from traditional mode
# class Mode_(Operation):
    # @classmethod
    # def create(cls, x: list[torch.Tensor], error: float) -> 'Mode':
    #     x_1d = to_1d(x[0])
    #     #  Mode has no result_error, just num_error which is the
    #     # deviation that two numbers are considered the same. (Make sense because
    #     # if some dataset has all different data, mode will be trivial if this is not the case)
    #     # This value doesn't depend on any scale, but on the dataset itself, and the intention
    #     # the evaluator. For example 0.01 means that data is counted as the same within 1% value range.

    #     # If wanting the strict definition of Mode, can just put this error to be 0
    #     result = torch.tensor(mode_within(x_1d, error))
    #     return cls(result, error)

    # def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
    #     # Assume x is [1, n, 1]
    #     x = x[0]
    #     size = x.size()[1]
    #     count_equal = torch.sum((torch.abs(x-self.result)<=torch.abs(self.error*self.result)).float())
    #     _result = torch.tensor([
    #         torch.sum((torch.abs(x-ele[0])<=torch.abs(self.error*ele[0])).float())<= count_equal
    #         for ele in x[0]
    #     ], dtype = torch.float32)
    #     return torch.sum(_result) == size


class Mode(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Mode':
        if precal_witness is None:
            x_1d = to_1d(x[0])
            x_1d = x_1d[x_1d!=MagicNumber]
            # Here is traditional definition of Mode, can just put this num_error to be 0
            result = torch.tensor(mode_within(x_1d, 0))
            return cls(result, error)
        else:
            if op_dict is None:
                return cls(torch.tensor(precal_witness['Mode_0'][0]), error)
            elif 'Mode' not in op_dict:
                return cls(torch.tensor(precal_witness['Mode_0'][0]), error)
            else:
                return cls(torch.tensor(precal_witness['Mode_'+str(op_dict['Mode'])][0]), error)
     

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [n, 1]
        x = x[0]
        min_x = torch.min(x)
        old_size = x.size()[0]
        x = torch.where(x==MagicNumber, min_x-1, x)
        count_equal = torch.sum(torch.where(x==self.result, 1.0, 0.0))

        count_check = 0
        for ele in x:
            bool1 = torch.sum(torch.where(x==ele[0], 1.0, 0.0))<=count_equal
            bool2 = ele[0]==min_x-1
            count_check += torch.logical_or(bool1, bool2)
        return count_check ==old_size


class PStdev(Operation):
    def __init__(self, x: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
            result = torch.sqrt(torch.var(x_1d, correction = 0))
            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['PStdev_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PStdev_0'][1]), requires_grad=False)
            elif 'PStdev' not in op_dict:
                super().__init__(torch.tensor(precal_witness['PStdev_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PStdev_0'][1]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['PStdev_'+str(op_dict['PStdev'])][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PStdev_'+str(op_dict['PStdev'])][1]), requires_grad=False)


    @classmethod
    def create(cls, x: list[torch.Tensor], error: float, precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'PStdev':
        return cls(x[0], error, precal_witness, op_dict)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.data_mean)
        return torch.logical_and(
            torch.abs(torch.sum((x_adj_mean)*(x_adj_mean))-self.result*self.result*size)<=torch.abs(2*self.error*self.result*self.result*size),x_mean_cons
        )


class PVariance(Operation):
    def __init__(self, x: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
            result = torch.var(x_1d, correction = 0)
            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['PVariance_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PVariance_0'][1]), requires_grad=False)
            elif 'PVariance' not in op_dict:
                super().__init__(torch.tensor(precal_witness['PVariance_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PVariance_0'][1]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['PVariance_'+str(op_dict['PVariance'])][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['PVariance_'+str(op_dict['PVariance'])][1]), requires_grad=False)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'PVariance':
        return cls(x[0], error, precal_witness, op_dict)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.data_mean)
        return torch.logical_and(
            torch.abs(torch.sum((x_adj_mean)*(x_adj_mean))-self.result*size)<=torch.abs(self.error*self.result*size), x_mean_cons
        )



class Stdev(Operation):
    def __init__(self, x: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
            result = torch.sqrt(torch.var(x_1d, correction = 1))
            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['Stdev_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Stdev_0'][1]), requires_grad=False)
            elif 'Stdev' not in op_dict:
                super().__init__(torch.tensor(precal_witness['Stdev_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Stdev_0'][1]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['Stdev_'+str(op_dict['Stdev'])][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Stdev_'+str(op_dict['Stdev'])][1]), requires_grad=False)


    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Stdev':
        return cls(x[0], error, precal_witness, op_dict)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.data_mean)
        return torch.logical_and(
            torch.abs(torch.sum((x_adj_mean)*(x_adj_mean))-self.result*self.result*(size - 1))<=torch.abs(2*self.error*self.result*self.result*(size - 1)), x_mean_cons
        )


class Variance(Operation):
    def __init__(self, x: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
            result = torch.var(x_1d, correction = 1)
            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['Variance_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Variance_0'][1]), requires_grad=False)         
            elif 'Variance' not in op_dict:
                super().__init__(torch.tensor(precal_witness['Variance_0'][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Variance_0'][1]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['Variance_'+str(op_dict['Variance'])][0]), error)
                self.data_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Variance_'+str(op_dict['Variance'])][1]), requires_grad=False)


    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Variance':
        return cls(x[0], error, precal_witness, op_dict)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.data_mean)
        return torch.logical_and(
            torch.abs(torch.sum((x_adj_mean)*(x_adj_mean))-self.result*(size - 1))<=torch.abs(self.error*self.result*(size - 1)), x_mean_cons
        )




class Covariance(Operation):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None: 
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            y_1d = to_1d(y)
            y_1d = y_1d[y_1d!=MagicNumber]
            x_1d_list = x_1d.tolist()
            y_1d_list = y_1d.tolist()

            self.x_mean = torch.nn.Parameter(data=torch.tensor(statistics.mean(x_1d_list), dtype = torch.float32), requires_grad=False)
            self.y_mean = torch.nn.Parameter(data=torch.tensor(statistics.mean(y_1d_list), dtype = torch.float32), requires_grad=False)
            result = torch.tensor(statistics.covariance(x_1d_list, y_1d_list), dtype = torch.float32)

            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['Covariance_0'][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_0'][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_0'][2]), requires_grad=False)  
            elif 'Covariance' not in op_dict:
                super().__init__(torch.tensor(precal_witness['Covariance_0'][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_0'][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_0'][2]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['Covariance_'+str(op_dict['Covariance'])][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_'+str(op_dict['Covariance'])][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Covariance_'+str(op_dict['Covariance'])][2]), requires_grad=False)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Covariance':
        return cls(x[0], x[1], error, precal_witness, op_dict)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
        x, y = args[0], args[1]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        y_fil_0 = torch.where(y==MagicNumber, 0.0, y)
        size_x = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        size_y = torch.sum(torch.where(y!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size_x*(self.x_mean))<=torch.abs(self.error*self.x_mean*size_x)
        y_mean_cons = torch.abs(torch.sum(y_fil_0)-size_y*(self.y_mean))<=torch.abs(self.error*self.y_mean*size_y)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.x_mean)
        y_adj_mean = torch.where(y==MagicNumber, 0.0, y-self.y_mean)

        return torch.logical_and(
            torch.logical_and(size_x==size_y,torch.logical_and(x_mean_cons,y_mean_cons)),
            torch.abs(torch.sum((x_adj_mean)*(y_adj_mean))-(size_x-1)*self.result)<=torch.abs(self.error*self.result*(size_x-1))
        )

# refer other constraints to correlation function, not put here since will be repetitive
def stdev_for_corr(x_adj_mean:torch.Tensor, size_x:torch.Tensor, x_std: torch.Tensor, error: float) -> torch.Tensor:
    return (
            torch.abs(torch.sum((x_adj_mean)*(x_adj_mean))-x_std*x_std*(size_x - 1))<=torch.abs(2*error*x_std*x_std*(size_x - 1))
        , x_std)
# refer other constraints to correlation function, not put here since will be repetitive
def covariance_for_corr(x_adj_mean: torch.Tensor,y_adj_mean: torch.Tensor,size_x:torch.Tensor, cov: torch.Tensor,  error: float) -> torch.Tensor:
        return (
            torch.abs(torch.sum((x_adj_mean)*(y_adj_mean))-(size_x-1)*cov)<=torch.abs(error*cov*(size_x-1))
        , cov)


class Correlation(Operation):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, error: float, precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1d = to_1d(x)
            x_1d = x_1d[x_1d!=MagicNumber]
            y_1d = to_1d(y)
            y_1d = y_1d[y_1d!=MagicNumber]
            x_1d_list = x_1d.tolist()
            y_1d_list = y_1d.tolist()
            self.x_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
            self.y_mean = torch.nn.Parameter(data=torch.mean(y_1d), requires_grad = False)
            self.x_std = torch.nn.Parameter(data=torch.sqrt(torch.var(x_1d, correction = 1)), requires_grad = False)
            self.y_std = torch.nn.Parameter(data=torch.sqrt(torch.var(y_1d, correction = 1)), requires_grad=False)
            self.cov = torch.nn.Parameter(data=torch.tensor(statistics.covariance(x_1d_list, y_1d_list), dtype = torch.float32), requires_grad=False)
            result = torch.tensor(statistics.correlation(x_1d_list, y_1d_list), dtype = torch.float32)

            super().__init__(result, error)
        else:
            if op_dict is None:
                super().__init__(torch.tensor(precal_witness['Correlation_0'][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][2]), requires_grad=False)
                self.x_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][3]), requires_grad=False)
                self.y_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][4]), requires_grad=False)
                self.cov = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][5]), requires_grad=False)       
            elif 'Correlation' not in op_dict:
                super().__init__(torch.tensor(precal_witness['Correlation_0'][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][2]), requires_grad=False)
                self.x_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][3]), requires_grad=False)
                self.y_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][4]), requires_grad=False)
                self.cov = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_0'][5]), requires_grad=False)
            else:
                super().__init__(torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][0]), error)
                self.x_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][1]), requires_grad=False)
                self.y_mean = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][2]), requires_grad=False)
                self.x_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][3]), requires_grad=False)
                self.y_std = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][4]), requires_grad=False)
                self.cov = torch.nn.Parameter(data = torch.tensor(precal_witness['Correlation_'+str(op_dict['Correlation'])][5]), requires_grad=False)


    @classmethod
    def create(cls, args: list[torch.Tensor], error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Correlation':
        return cls(args[0], args[1], error, precal_witness, op_dict)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
        x, y = args[0], args[1]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        y_fil_0 = torch.where(y==MagicNumber, 0.0, y)
        size_x = torch.sum(torch.where(x!=MagicNumber, 1.0, 0.0))
        size_y = torch.sum(torch.where(y!=MagicNumber, 1.0, 0.0))
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size_x*(self.x_mean))<=torch.abs(self.error*self.x_mean*size_x)
        y_mean_cons = torch.abs(torch.sum(y_fil_0)-size_y*(self.y_mean))<=torch.abs(self.error*self.y_mean*size_y)
        x_adj_mean = torch.where(x==MagicNumber, 0.0, x-self.x_mean)
        y_adj_mean = torch.where(y==MagicNumber, 0.0, y-self.y_mean)

        miscel_cons = torch.logical_and(size_x==size_y, torch.logical_and(x_mean_cons, y_mean_cons))
        bool1, cov = covariance_for_corr(x_adj_mean,y_adj_mean,size_x, self.cov, self.error)
        bool2, x_std = stdev_for_corr( x_adj_mean, size_x, self.x_std, self.error)
        bool3, y_std = stdev_for_corr( y_adj_mean, size_y, self.y_std, self.error)
        # this is correlation constraint
        bool4 = torch.abs(cov - self.result*x_std*y_std)<=torch.abs(self.error*cov)
        return torch.logical_and(torch.logical_and(torch.logical_and(bool1, bool2),torch.logical_and(bool3, bool4)), miscel_cons)


def stacked_x(args: list[float]):
    return np.column_stack((*args, np.ones_like(args[0])))


class Regression(Operation):
    def __init__(self, xs: list[torch.Tensor], y: torch.Tensor, error: float,  precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None):
        if precal_witness is None:
            x_1ds = [to_1d(i) for i in xs]
            fil_x_1ds=[]
            for x_1 in x_1ds:
                fil_x_1ds.append((x_1[x_1!=MagicNumber]).tolist())
            x_1ds = fil_x_1ds

            y_1d = to_1d(y)
            y_1d = (y_1d[y_1d!=MagicNumber]).tolist()

            x_one = stacked_x(x_1ds)
            result_1d = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_one.transpose(), x_one)), x_one.transpose()), y_1d)
            # result = torch.tensor(result_1d, dtype = torch.float32).reshape(1, -1, 1)
            result = torch.tensor(result_1d, dtype = torch.float32).reshape(-1,1)
            super().__init__(result, error)
            # print('result regression: ', result)
        else:
            if op_dict is None:
                # result = torch.tensor(precal_witness['Regression_0']).reshape(1,-1,1)
                result = torch.tensor(precal_witness['Regression_0']).reshape(-1,1)
            elif 'Regression' not in op_dict:
                # result = torch.tensor(precal_witness['Regression_0']).reshape(1,-1,1)
                result = torch.tensor(precal_witness['Regression_0']).reshape(-1,1)
            else:
                # result = torch.tensor(precal_witness['Regression_'+str(op_dict['Regression'])]).reshape(1,-1,1)
                result = torch.tensor(precal_witness['Regression_'+str(op_dict['Regression'])]).reshape(-1,1)

            # for ele in precal_witness['Regression']:
            #     precal_witness_arr.append(torch.tensor(ele))
            # print('resultopppp: ', result)
            super().__init__(result,error)
            

    @classmethod
    def create(cls, args: list[torch.Tensor], error: float, precal_witness:Optional[dict] = None, op_dict:Optional[dict[str,int]] = None) -> 'Regression':
        xs = args[:-1]
        y = args[-1]
        return cls(xs, y, error, precal_witness, op_dict)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
         # infer y from the last parameter
        y = args[-1]
        y = torch.where(y==MagicNumber,0.0, y)
        x_one = torch.cat((*args[:-1], torch.ones_like(args[0])), dim = 1)
        x_one = torch.where((x_one[:,0] ==MagicNumber).unsqueeze(-1), torch.tensor([0.0]*x_one.size()[1]), x_one)
        x_t = torch.transpose(x_one, 0, 1)

        left = x_t @ x_one @ self.result - x_t @ y
        right = self.error*x_t @ y
        abs_left = torch.where(left>=0, left, -left)
        abs_right = torch.where(right>=0, right, -right)
        return torch.where(torch.sum(torch.where(abs_left<=abs_right, 1.0, 0.0))==torch.tensor(2.0), 1.0, 0.0)

