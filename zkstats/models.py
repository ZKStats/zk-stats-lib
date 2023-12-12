from typing import Any
from abc import ABC, abstractmethod
from torch import nn
import torch


class BaseZKStatsModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X: Any) -> Any:
        """
        :param X: a tensor of shape (1, n, 1)
        :return: a tuple of (bool, float)
        """


class NoDivisionModel(BaseZKStatsModel):
    def __init__(self):
        super().__init__()
        # w represents mean in this case

    @abstractmethod
    def prepare(expected_output: Any):
        ...

    @abstractmethod
    def forward(self, X: Any) -> tuple[float, float]:
        # some expression of tolerance to error in the inference
        # must have w first!
        ...

class MeanModel(NoDivisionModel):
    def __init__(self):
        super().__init__()

    def prepare(self, X: Any):
        expected_output = torch.mean(X[0])
        # w represents mean in this case
        self.w = nn.Parameter(data = expected_output, requires_grad = False)

    def forward(self, X: Any) -> tuple[float, float]:
        # some expression of tolerance to error in the inference
        # must have w first!
        return (torch.abs(torch.sum(X)-X.size()[1]*(self.w))<0.01*X.size()[1]*(self.w), self.w)
