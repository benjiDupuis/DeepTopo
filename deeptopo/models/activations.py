import numpy as np
import torch

# Some normalized activation functions


def normalized_relu(x):
    return np.sqrt(2.)*torch.relu(x)


class NormalizedSin(object):

    def __init__(self, a: float):
        self.norm = np.sqrt(2./(1 - np.exp(-2.*a*a)))
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.sin(self.omega * x)


class NormalizedCos(object):

    def __init__(self, a: float):
        self.norm = np.sqrt(2./(1 + np.exp(-2.*a*a)))
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.cos(self.omega * x)


class NormalizedExp(object):

    def __init__(self, a: float):
        self.norm = np.exp(a**2)
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.exp(self.omega * x)
