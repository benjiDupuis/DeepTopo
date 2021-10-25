from typing import Tuple

import torch

from deeptopo.models.embeddings import Embedding
from deeptopo.topoptim.topopt2D import Topopt2D
from deeptopo.topoptim.loadcase import LoadCase


class DeepTopo(Topopt2D):

    method = None
    ft: int = 2

    def __init__(self, shape, embedding, net, loadcase, volfrac):

    def __call__()
    # TODO: cf drive pour le training ici
    # Idée: tout le monde (net et embedding prennent la shape en argument ici, comme ça pas de confusion)
