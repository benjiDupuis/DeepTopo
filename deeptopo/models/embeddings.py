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
        grid = rearrange(grid, "a b c -> c b a")
        grid = rearrange(grid, "h w d -> (h w) d")
        grid.to(device)
        return dr*grid

    @torch.no_grad()
    def varphi(self,  x: torch.Tensor, shape: Tuple[int]) -> torch.Tensor:
        pass

    @torch.no_grad()
    def __call__(self, shape, dr: Optional[float] = 1.) -> torch.Tensor:
        grid = Embedding2D.make_grid(shape, dr=dr)
        return self.varphi(grid, shape)


class GaussianEmbedding(Embedding2D):

    def __init__(self, size: int, ell: float = 1.):
        self.size = size
        self.ell = ell
        self.w0 = torch.normal(0., 1./self.ell, (2, self.size))

    @torch.no_grad()
    def varphi(self, x: torch.Tensor, *args) -> torch.Tensor:
        return np.sqrt(2.)*torch.sin(torch.mm(x, self.w0) + np.pi/4.)


class TorusEmbedding(Embedding2D):

    def __init__(self, delta: Optional[float] = None,
                 r: Optional[float] = np.sqrt(2.)):
        self.r = r
        self.size = 4
        self.delta = delta

    @torch.no_grad()
    def varphi(self, x: torch.Tensor, shape: Tuple[int]) -> torch.Tensor:
        if self.delta is None:
            delta = np.pi/(2.*max(shape[0], shape[1]))
        else:
            delta = self.delta

        assert len(x.shape) == 2, "grid has wrong shape"
        return self.r*torch.cat([torch.cos(delta*x[:, 0]).unsqueeze(1),
                                 torch.sin(delta*x[:, 1]).unsqueeze(1),
                                 torch.cos(delta*x[:, 2]).unsqueeze(1),
                                 torch.sin(delta*x[:, 3]).unsqueeze(1)], dim=1)
