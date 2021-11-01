from typing import Tuple

import numpy as np

from deeptopo.topoptim.loadcase import LoadCase


class MichellStructure(LoadCase):

    def __init__(self, shape: Tuple[int]):
        super(MichellStructure, self).__init__(shape)
        self.add_force((self.shape[0] - 1, self.shape[1]//2), np.array([0., -1.]))
        self.fixed[0, (self.shape[1]//4):(3*self.shape[1]//4), :] = True


class Cantilever(LoadCase):

    def __init__(self, shape: Tuple[int]):
        super(MichellStructure, self).__init__(shape)
        self.add_force((self.shape[0] - 1, self.shape[1] - 1), np.array([0., -1.]))
        self.fixed[0, :, :] = True
