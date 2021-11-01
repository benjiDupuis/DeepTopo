from pathlib import Path

import cv2
import fire
import numpy as np
import torch

from deeptopo.training.trainer import DeepTopo
from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.topoptim.loadcase import MichellStructure
from deeptopo.models.activations import normalized_relu


def main(iter: int = 200, output_dir="results"):
    '''
    Example of DNN-based topology optimization
    '''
    output_dir = Path(output_dir) / "michell_structure.png"

    loadcase = MichellStructure((100, 50))
    optimizer = DeepTopo(loadcase,
                         GaussianEmbedding(1000, ell=3.),
                         FCNN([1000], beta=0.7, activation=normalized_relu),
                         0.3)

    field = optimizer(iter)
    cv2.imwrite(str(output_dir), 255 - (255*field).astype(np.uint8))


if __name__ == "__main__":
    fire.Fire(main)
