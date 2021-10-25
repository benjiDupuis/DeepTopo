from pathlib import Path

import cv2
import fire
import numpy as np
import torch

from deeptopo.training.trainer import DeepTopo
from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.topoptim.loadcase import MichellStructure


def main(iter: int = 200, output_dir="results"):
    '''
    Example of DNN-based topology optimization
    '''
    output_dir = Path(output_dir) / "michell_structure.png"

    loadcase = MichellStructure((100, 50))
    optimizer = DeepTopo(loadcase,
                         FCNN(GaussianEmbedding(200, ell=30.),
                              [50, 50, 50], activation=torch.relu),
                         0.4)

    field = optimizer(iter)
    cv2.imwrite(str(output_dir), 255 - (255*field).astype(np.uint8))


if __name__ == "__main__":
    fire.Fire(main)
