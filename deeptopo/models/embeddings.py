from typing import Tuple

import numpy as np
import torch
from einops import rearrange


class Embedding:

    size: int

    @staticmethod
    def make_grid(shape: Tuple[int]) -> torch.Tensor:
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
        return grid

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def __call__(self, shape) -> torch.Tensor:
        grid = Embedding.make_grid(shape)
        return self.varphi(grid)


class GaussianEmbedding(Embedding):

    def __init__(self, size: int, ell: float = 1.):
        self.size = size
        self.ell = ell
        self.w0 = torch.normal(0., 1./self.ell, (2, self.size))

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:
        return np.sqrt(2.)*torch.sin(torch.mm(x, self.w0) + np.pi/4.)


# class TorusEmbedding(Embedding):

#     @staticmethod
#     def input(shape)
