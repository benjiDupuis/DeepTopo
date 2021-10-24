from typing import Tuple

import numpy as np
import torch
from einops import rearrange


class Embedding:

    size: int

    def __init__(self, shape: Tuple[int]):
        '''
        Initialization of the grid coordinates
        '''
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        x, y = torch.meshgrid([torch.arange(shape[0]),
                               torch.arange(shape[1])])
        self.shape = shape
        grid = torch.stack((x, y)).type(torch.FloatTensor)
        grid = rearrange(grid, "a b c -> c b a")
        self.grid = rearrange(grid, "h w d -> (h w) d")
        self.grid.to(device)

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        return self.varphi(self.grid)


class GaussianEmbedding(Embedding):

    def __init__(self, shape: Tuple[int], size: int, ell: float = 1.):
        super(GaussianEmbedding, self).__init__(shape)
        self.size = size
        self.ell = ell
        self.w0 = torch.normal(0., 1./self.ell, (2, self.size))

    @torch.no_grad()
    def varphi(self, x: torch.Tensor) -> torch.Tensor:
        return np.sqrt(2.)*torch.sin(torch.mm(self.grid, self.w0) + np.pi/4.)


# class TorusEmbedding(Embedding):

#     @staticmethod
#     def input(shape)
