"""
Definition of normalized activation functions \n
Their respective dual and derivative dual functions are also implemented \n
Those last functions can be used to compute the theoretical NTK of the network
"""
import numpy as np
import torch


class NormalizedActivation():
    """
    Generic class for normalized activations
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def dual(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def dual_derivative(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NormalizedReLu(NormalizedActivation):

    def __call__(self, x):
        return np.sqrt(2.)*torch.relu(x)

    @torch.no_grad()
    def dual(self, rho: torch.Tensor) -> torch.Tensor:
        return rho - rho*(torch.arccos(rho) - torch.sqrt(1 - rho.pow(2)))/torch.pi

    @torch.no_grad()
    def dual_derivative(self, rho: torch.Tensor) -> torch.Tensor:
        return 1. - torch.arccos(rho)/torch.pi


class NormalizedSin(NormalizedActivation):

    def __init__(self, a: float):
        self.norm = np.sqrt(2./(1 - np.exp(-2.*a*a)))
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.sin(self.omega * x)

    @torch.no_grad()
    def dual(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.sinh((self.omega**2)*rho)/np.sinh(self.omega**2)

    @torch.no_grad()
    def dual_derivative(self, rho: torch.Tensor) -> torch.Tensor:
        return (self.omega**2)*torch.cosh((self.omega**2)*rho)/np.sinh(self.omega**2)


class NormalizedCos(NormalizedActivation):

    def __init__(self, a: float):
        self.norm = np.sqrt(2./(1 + np.exp(-2.*a*a)))
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.cos(self.omega * x)

    @torch.no_grad()
    def dual(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.cosh((self.omega**2)*rho)/np.cosh(self.omega**2)

    @torch.no_grad()
    def dual_derivative(self, rho: torch.Tensor) -> torch.Tensor:
        return (self.omega**2)*torch.sinh((self.omega**2)*rho)/np.cosh(self.omega**2)


class NormalizedExp(NormalizedActivation):

    def __init__(self, a: float):
        self.norm = np.exp(a**2)
        self.omega = a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.exp(self.omega * x)

    @torch.no_grad()
    def dual(self, rho: torch.Tensor) -> torch.Tensor:
        return super().dual(rho)

    @torch.no_grad()
    def dual_derivative(self, rho: torch.Tensor) -> torch.Tensor:
        return super().dual_derivative(rho)
