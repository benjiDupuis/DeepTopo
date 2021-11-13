from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from deeptopo.models.embeddings import Embedding2D
from deeptopo.models.networks import FCNN
from deeptopo.models.activations import NormalizedActivation


class TheoreticalNTK():
    """
    Implementation of the theoretical limiting NTK as described in the paper \n
    This class is made to be applicable with embeddings
    and networks defined in this repository
    """

    def __init__(self,
                 net: FCNN,
                 embedding: Embedding2D):

        self.beta = net.beta
        self.layers = net.layers
        self.L = net.L
        self.embedding = embedding
        self.act = net.act

        # We assert that the activation has dual functions
        assert isinstance(self.act, NormalizedActivation), \
            "activation function should have duals implemented"
        assert 0. <= self.beta and self.beta <= 1., "beta should be in [0, 1]"

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(0) if x.ndim == 1 else x
        y = x.unsqueeze(0) if y.ndim == 1 else y

        sigma = self.embedding.kernel(x, y)
        ntk = sigma

        for _ in tqdm(range(self.L - 1)):
            sigma = self.beta**2 + (1. - self.beta**2)*self.act.dual(sigma)
            sigma_dot = (1. - self.beta**2)*self.act.dual_derivative(sigma)
            ntk = ntk*sigma_dot + sigma

        return ntk

    @torch.no_grad()
    def one_line_ntk(self, shape: Tuple[int],
                     index: Optional[int] = None,
                     dr: Optional[float] = 1.):

        index = shape[1]*(1 + shape[0])//2 if index is None else index

        grid = Embedding2D.make_grid(shape)
        base_point = grid[index]
        base_point_grid = torch.stack([base_point for _ in range(grid.shape[0])])

        ntk = self.__call__(base_point_grid, grid)
        ntk = ntk.reshape(shape[0], shape[1])

        return ntk

    def visualize_one_line_ntk(self, shape: Tuple[int],
                               output_dir: str,
                               index: Optional[int] = None,
                               dr: Optional[float] = 1.):

        ntk = self.one_line_ntk(shape, index, dr)
        if torch.max(ntk) > 1.e-6:
            ntk = ntk/torch.max(ntk)
        ntk = ntk.cpu().data.numpy()

        output_path = Path(output_dir) / "theoretical_ntk.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), 255 - (255*ntk).astype(np.uint8))
