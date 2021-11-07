from pathlib import Path

import cv2
import fire
import numpy as np

from deeptopo.training.trainer import DeepTopo
from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.topoptim.loadcase_zoo import MichellStructure
from deeptopo.models.activations import normalized_relu


def main(iter: int = 200, output_dir="results",
         up_sampling_factor: float = 4.):
    """
    Example of DNN-based topology optimization
    """
    output_path = Path(output_dir) / "michell_structure.png"

    loadcase = MichellStructure((100, 50))
    optimizer = DeepTopo(loadcase,
                         GaussianEmbedding(1000, ell=10.),
                         FCNN([1000], beta=0.9, activation=normalized_relu),
                         0.4)

    field = optimizer(iter)
    cv2.imwrite(str(output_path), 255 - (255*field).astype(np.uint8))

    output_path = Path(output_dir) / "michell_structure_up_sampled.png"
    cv2.imwrite(str(output_path),
                255 - 255*optimizer.up_sampling(up_sampling_factor).astype(np.uint8))


if __name__ == "__main__":
    fire.Fire(main)
