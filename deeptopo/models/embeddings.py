from typing import Tuple, Optional

import numpy as np
import torch
from einops import rearrange


class Embedding2D:

    size: int

    @staticmethod
    def make_grid(shape: Tuple[int], dr: Optional[float] = 1.) -> torch.Tensor:
        '''
        Initialization of the grid coordinates
        '''
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        x, y = torch.meshgrid([torch.arange(shape[0]),
                               torch.arange(shape[1])])
        grid = torch.stack((x, y)).type(torch.FloatTensor)
        grid = rearrange(grid, "a b c -> b c a")
        grid = rearrange(grid, "h w d -> (h w) d")
        grid.to(device)
        return dr*grid

    @torch.no_grad()
    def varphi(self,  x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def kernel(self, **args) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, shape, dr: Optional[float] = 1.) -> torch.Tensor:
        grid = Embedding2D.make_grid(shape, dr=dr)
        return self.varphi(grid)


class GaussianEmbedding(Embedding2D):

    def __init__(self, size: int, ell: float = 1.):
        self.size = size
        self.ell = ell
        self.w0 = torch.normal(0., 1./self.ell, (2, self.size))

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:
        return np.sqrt(2.)*torch.sin(torch.mm(x, self.w0) + np.pi/4.)

    @torch.no_grad()
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x and y: shape (B, d) where d is the dimension of euclidean space \n
        :output: tensor of shape B
        """
        assert x.ndim == 2, "x tensor should be of dim 2"
        assert y.ndim == 2, "y tensor should be of dim 2"

        return torch.exp(-(x - y).pow(2).sum(dim=1)/(2.*(self.ell**2)))


class TorusEmbedding(Embedding2D):

    def __init__(self, delta: Optional[float] = None,
                 r: Optional[float] = np.sqrt(2.),
                 shape: Optional[Tuple[int]] = None):
        self.r = r
        self.size = 4
        self.delta = delta
        self.shape = shape

    @property
    def default_delta(s: Optional[Tuple[int]] = None) -> float:
        if s is None:
            print("WARNING: This is a torus embedding but neither delta \
                not shape have been defined, delta has been set to 1 by default")
            return 1.
        else:
            return np.pi/(2.*max(s[0], s[1]))

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:

        assert len(x.shape) == 2, "grid has wrong shape"
        return self.r*torch.cat([torch.cos(self.delta*x[:, 0]).unsqueeze(1),
                                 torch.sin(self.delta*x[:, 1]).unsqueeze(1),
                                 torch.cos(self.delta*x[:, 2]).unsqueeze(1),
                                 torch.sin(self.delta*x[:, 3]).unsqueeze(1)], dim=1)

    @torch.no_grad()
    def kernel(self, x: torch.Tensor, y: torch.Tensor, shape: Tuple[int])\
            -> torch.Tensor():

        assert x.ndim == 2, "x tensor should have two dimensions"
        assert y.ndim == 2, "y tensor should have two dimensions"

        return 0.5*torch.cos(self.delta*(x - y)).sum(dim=1)
