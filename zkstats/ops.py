from abc import ABC, abstractmethod, abstractclassmethod
import statistics

import numpy as np
import torch

# boolean: either 1.0 or 0.0
IsResultPrecise = torch.Tensor
MagicNumber = 9999999


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


class Where(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Where':
        # here error is trivial, but here to conform to other functions
        return cls(torch.where(x[0],x[1], MagicNumber ),error)
    def ezkl(self, x:list[torch.Tensor]) -> IsResultPrecise:
        bool_array = torch.logical_or(x[1]==self.result, torch.logical_and(torch.logical_not(x[0]), self.result==MagicNumber))
        # print('sellll: ', self.result)
        return torch.sum(bool_array.float())==x[1].size()[1]


class Mean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Mean':
        # support where statement, hopefully we can use 'nan' once onnx.isnan() is supported
        return cls(torch.mean(x[0][x[0]!=MagicNumber]), error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        size = torch.sum((x!=MagicNumber).float())
        x = torch.where(x==MagicNumber, 0.0, x)
        return torch.abs(torch.sum(x)-size*self.result)<=torch.abs(self.error*self.result*size)


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
        x_1d = x_1d[x_1d!=MagicNumber]
        super().__init__(torch.tensor(np.median(x_1d)), error)
        sorted_x = np.sort(x_1d)
        len_x = len(x_1d)
        self.lower = torch.nn.Parameter(data = torch.tensor(sorted_x[int(len_x/2)-1], dtype = torch.float32), requires_grad=False)
        self.upper = torch.nn.Parameter(data = torch.tensor(sorted_x[int(len_x/2)], dtype = torch.float32), requires_grad=False)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Median':
        return cls(x[0], error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        old_size = x.size()[1]
        size = torch.sum((x!=MagicNumber).float())
        min_x = torch.min(x)
        x = torch.where(x==MagicNumber,min_x-1, x)

        # since within 1%, we regard as same value
        count_less = torch.sum((x < self.result).float())-(old_size-size)
        count_equal = torch.sum((x==self.result).float())
        half_size = torch.floor(torch.div(size, 2))

        less_cons = count_less<half_size+size%2
        more_cons = count_less+count_equal>half_size

        # For count_equal == 0
        lower_exist = torch.sum((x==self.lower).float())>0
        lower_cons = torch.sum((x>self.lower).float())==half_size
        upper_exist = torch.sum((x==self.upper).float())>0
        upper_cons = torch.sum((x<self.upper).float())==half_size
        bound = count_less== half_size
        # 0.02 since 2*0.01
        bound_avg = (torch.abs(self.lower+self.upper-2*self.result)<=torch.abs(2*self.error*self.result))

        median_in_cons = torch.logical_and(less_cons, more_cons)
        median_out_cons = torch.logical_and(torch.logical_and(bound, bound_avg), torch.logical_and(torch.logical_and(lower_cons, upper_cons), torch.logical_and(lower_exist, upper_exist)))
        return torch.where(count_equal==0, median_out_cons, median_in_cons)


class GeometricMean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'GeometricMean':
        x_1d = to_1d(x[0])
        x_1d = x_1d[x_1d!=MagicNumber]
        result = torch.exp(torch.mean(torch.log(x_1d)))
        return cls(result, error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [1, n, 1]
        x = x[0]
        size = torch.sum((x!=MagicNumber).float())
        x = torch.where(x==MagicNumber, 1.0, x)
        return torch.abs((torch.log(self.result)*size)-torch.sum(torch.log(x)))<=size*torch.log(torch.tensor(1+self.error))


class HarmonicMean(Operation):
    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'HarmonicMean':
        x_1d = to_1d(x[0])
        x_1d = x_1d[x_1d!=MagicNumber]
        result = torch.div(1.0,torch.mean(torch.div(1.0, x_1d)))
        return cls(result, error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [1, n, 1]
        x = x[0]
        size = torch.sum((x!=MagicNumber).float())
        # just make it really big so that 1/x goes to zero for element that gets filtered out
        x = torch.where(x==MagicNumber, x*x, x)
        return torch.abs((self.result*torch.sum(torch.div(1.0, x))) - size)<=torch.abs(self.error*size)


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
    def create(cls, x: list[torch.Tensor], error: float) -> 'Mode':
        x_1d = to_1d(x[0])
        x_1d = x_1d[x_1d!=MagicNumber]
        # Here is traditional definition of Mode, can just put this num_error to be 0
        result = torch.tensor(mode_within(x_1d, 0))
        return cls(result, error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        # Assume x is [1, n, 1]
        x = x[0]
        min_x = torch.min(x)
        old_size = x.size()[1]
        x = torch.where(x==MagicNumber, min_x-1, x)
        count_equal = torch.sum((x==self.result).float())
        result = torch.tensor([torch.logical_or(torch.sum((x==ele[0]).float())<=count_equal, min_x-1 ==ele[0]) for ele in x[0]])
        return torch.sum(result) == old_size


class PStdev(Operation):
    def __init__(self, x: torch.Tensor, error: float):
        x_1d = to_1d(x)
        x_1d = x_1d[x_1d!=MagicNumber]
        self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
        result = torch.sqrt(torch.var(x_1d, correction = 0))
        super().__init__(result, error)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'PStdev':
        return cls(x[0], error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum((x!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_fil_mean = torch.where(x==MagicNumber, self.data_mean, x)
        return torch.logical_and(
            torch.abs(torch.sum((x_fil_mean-self.data_mean)*(x_fil_mean-self.data_mean))-self.result*self.result*size)<=torch.abs(2*self.error*self.result*self.result*size),x_mean_cons
        )


class PVariance(Operation):
    def __init__(self, x: torch.Tensor, error: float):
        x_1d = to_1d(x)
        x_1d = x_1d[x_1d!=MagicNumber]
        self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
        result = torch.var(x_1d, correction = 0)
        super().__init__(result, error)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'PVariance':
        return cls(x[0], error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum((x!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_fil_mean = torch.where(x==MagicNumber, self.data_mean, x)
        return torch.logical_and(
            torch.abs(torch.sum((x_fil_mean-self.data_mean)*(x_fil_mean-self.data_mean))-self.result*size)<=torch.abs(self.error*self.result*size), x_mean_cons
        )



class Stdev(Operation):
    def __init__(self, x: torch.Tensor, error: float):
        x_1d = to_1d(x)
        x_1d = x_1d[x_1d!=MagicNumber]
        self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
        result = torch.sqrt(torch.var(x_1d, correction = 1))
        super().__init__(result, error)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Stdev':
        return cls(x[0], error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum((x!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_fil_mean = torch.where(x==MagicNumber, self.data_mean, x)
        return torch.logical_and(
            torch.abs(torch.sum((x_fil_mean-self.data_mean)*(x_fil_mean-self.data_mean))-self.result*self.result*(size - 1))<=torch.abs(2*self.error*self.result*self.result*(size - 1)), x_mean_cons
        )


class Variance(Operation):
    def __init__(self, x: torch.Tensor, error: float):
        x_1d = to_1d(x)
        x_1d = x_1d[x_1d!=MagicNumber]
        self.data_mean = torch.nn.Parameter(data=torch.mean(x_1d), requires_grad=False)
        result = torch.var(x_1d, correction = 1)
        super().__init__(result, error)

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Variance':
        return cls(x[0], error)

    def ezkl(self, x: list[torch.Tensor]) -> IsResultPrecise:
        x = x[0]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        size = torch.sum((x!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size*(self.data_mean))<=torch.abs(self.error*self.data_mean*size)
        x_fil_mean = torch.where(x==MagicNumber, self.data_mean, x)
        return torch.logical_and(
            torch.abs(torch.sum((x_fil_mean-self.data_mean)*(x_fil_mean-self.data_mean))-self.result*(size - 1))<=torch.abs(self.error*self.result*(size - 1)), x_mean_cons
        )




class Covariance(Operation):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, error: float):
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

    @classmethod
    def create(cls, x: list[torch.Tensor], error: float) -> 'Covariance':
        return cls(x[0], x[1], error)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
        x, y = args[0], args[1]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        y_fil_0 = torch.where(y==MagicNumber, 0.0, y)
        size_x = torch.sum((x!=MagicNumber).float())
        size_y = torch.sum((y!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size_x*(self.x_mean))<=torch.abs(self.error*self.x_mean*size_x)
        y_mean_cons = torch.abs(torch.sum(y_fil_0)-size_y*(self.y_mean))<=torch.abs(self.error*self.y_mean*size_y)
        x_fil_mean = torch.where(x==MagicNumber, self.x_mean, x)
        # only x_fil_mean is enough, no need for y_fil_mean since it will multiply 0 anyway
        return torch.logical_and(
            torch.logical_and(size_x==size_y,torch.logical_and(x_mean_cons,y_mean_cons)),
            torch.abs(torch.sum((x_fil_mean-self.x_mean)*(y-self.y_mean))-(size_x-1)*self.result)<self.error*self.result*(size_x-1)
        )

# refer other constraints to correlation function, not put here since will be repetitive
def stdev_for_corr(x_fil_mean:torch.Tensor,size_x:torch.Tensor, x_std: torch.Tensor, x_mean: torch.Tensor, error: float) -> torch.Tensor:
    return (
            torch.abs(torch.sum((x_fil_mean-x_mean)*(x_fil_mean-x_mean))-x_std*x_std*(size_x - 1))<=torch.abs(2*error*x_std*x_std*(size_x - 1))
        , x_std)
# refer other constraints to correlation function, not put here since will be repetitive
def covariance_for_corr(x_fil_mean: torch.Tensor,y_fil_mean: torch.Tensor,size_x:torch.Tensor, size_y:torch.Tensor, cov: torch.Tensor, x_mean: torch.Tensor, y_mean: torch.Tensor, error: float) -> torch.Tensor:
        return (
            torch.abs(torch.sum((x_fil_mean-x_mean)*(y_fil_mean-y_mean))-(size_x-1)*cov)<error*cov*(size_x-1)
        , cov)


class Correlation(Operation):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, error: float):
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

    @classmethod
    def create(cls, args: list[torch.Tensor], error: float) -> 'Correlation':
        return cls(args[0], args[1], error)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
        x, y = args[0], args[1]
        x_fil_0 = torch.where(x==MagicNumber, 0.0, x)
        y_fil_0 = torch.where(y==MagicNumber, 0.0, y)
        size_x = torch.sum((x!=MagicNumber).float())
        size_y = torch.sum((y!=MagicNumber).float())
        x_mean_cons = torch.abs(torch.sum(x_fil_0)-size_x*(self.x_mean))<=torch.abs(self.error*self.x_mean*size_x)
        y_mean_cons = torch.abs(torch.sum(y_fil_0)-size_y*(self.y_mean))<=torch.abs(self.error*self.y_mean*size_y)
        x_fil_mean = torch.where(x==MagicNumber, self.x_mean, x)
        y_fil_mean = torch.where(y==MagicNumber, self.y_mean, y)

        miscel_cons = torch.logical_and(size_x==size_y, torch.logical_and(x_mean_cons, y_mean_cons))
        bool1, cov = covariance_for_corr(x_fil_mean,y_fil_mean,size_x, size_y, self.cov, self.x_mean, self.y_mean, self.error)
        bool2, x_std = stdev_for_corr( x_fil_mean, size_x, self.x_std, self.x_mean, self.error)
        bool3, y_std = stdev_for_corr( y_fil_mean, size_y, self.y_std, self.y_mean, self.error)
        bool4 = torch.abs(cov - self.result*x_std*y_std)<=self.error*cov
        return torch.logical_and(torch.logical_and(torch.logical_and(bool1, bool2),torch.logical_and(bool3, bool4)), miscel_cons)


def stacked_x(args: list[float]):
    return np.column_stack((*args, np.ones_like(args[0])))


class Regression(Operation):
    def __init__(self, xs: list[torch.Tensor], y: torch.Tensor, error: float):
        x_1ds = [to_1d(i) for i in xs]
        fil_x_1ds=[]
        for x_1 in x_1ds:
            fil_x_1ds.append((x_1[x_1!=MagicNumber]).tolist())
        x_1ds = fil_x_1ds

        y_1d = to_1d(y)
        y_1d = (y_1d[y_1d!=MagicNumber]).tolist()

        x_one = stacked_x(x_1ds)
        result_1d = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_one.transpose(), x_one)), x_one.transpose()), y_1d)
        result = torch.tensor(result_1d, dtype = torch.float32).reshape(1, -1, 1)
        print('result: ', result)
        super().__init__(result, error)

    @classmethod
    def create(cls, args: list[torch.Tensor], error: float) -> 'Regression':
        xs = args[:-1]
        y = args[-1]
        return cls(xs, y, error)

    def ezkl(self, args: list[torch.Tensor]) -> IsResultPrecise:
         # infer y from the last parameter
        y = args[-1]
        y = torch.where(y==MagicNumber, torch.tensor(0.0), y)
        x_one = torch.cat((*args[:-1], torch.ones_like(args[0])), dim=2)
        x_one = torch.where((x_one[:,:,0] ==MagicNumber).unsqueeze(-1), torch.tensor([0.0]*x_one.size()[2]), x_one)
        x_t = torch.transpose(x_one, 1, 2)
        return torch.sum(torch.abs(x_t @ x_one @ self.result - x_t @ y)) <= self.error * torch.sum(torch.abs(x_t @ y))

