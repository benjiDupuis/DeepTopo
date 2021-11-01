from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch.nn as nn

from deeptopo.models.embeddings import Embedding2D
from deeptopo.ntk.empirical_ntk import empirical_ntk_1D


def visualize_ntk(net: nn.Module,
                  shape: Tuple[int],
                  output_dir: str,
                  embedding: Optional[Embedding2D] = None):

    if embedding is None:
        batch_input = Embedding2D.make_grid(shape)
    else:
        batch_input = embedding(shape)
    ntk = empirical_ntk_1D(batch_input, net)
    ntk = ntk.reshape(shape[0], shape[1]).cpu().detach().data.numpy()

    ntk = ntk - np.min(ntk)
    if np.max(ntk) >= 1.e-6:
        ntk = ntk/np.max(ntk)
    ntk = 255*ntk.astype(np.uint8)

    output_path = Path(output_dir) / "ntk.png"
    cv2.imwrite(str(output_path), ntk)
