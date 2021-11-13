from pathlib import Path

import cv2
import fire
import numpy as np

from deeptopo.topoptim.loadcase_zoo import MichellStructure
from deeptopo.topoptim.topopt2D import Topopt2D


def main(iter: int = 50, output_dir="results"):
    """
    Example of SIMP/OC method
    """
    output_path = Path(output_dir) / "michell_structure.png"

    loadcase = MichellStructure((100, 50))
    optimizer = Topopt2D(loadcase, 0.4)

    field, _ = optimizer(iter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), 255 - (255*field).astype(np.uint8))


if __name__ == "__main__":
    fire.Fire(main)
