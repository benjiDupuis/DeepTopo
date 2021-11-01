import numpy as np
from typing import Union, Tuple


class LoadCase(object):
    '''
    Definition of a class for loadcases
    '''

    def __init__(self, shape: Tuple[int]):

        self.shape = (shape[0] + 1, shape[1] + 1)
        self.fixed = np.zeros((self.shape[0], self.shape[1], 2), dtype=np.bool8)
        self.forces = np.zeros((self.shape[0], self.shape[1], 2))

    def add_force(self, position: Tuple[int], vector: np.ndarray):
        '''
        add a force vector at the given position \n
        position: Tuple of size 2 (pos_x, pos_y) \n
        vector: force vector (size 2)
        '''
        self.forces[position[0], position[1], :] = vector

    def set_forces(self, f: np.ndarray):
        self.forces = f

    def add_fixed_point(self, position: Union[np.ndarray, Tuple[int]],
                        directions: list = [0, 1]):
        for d in directions:
            self.fixed[position[0], position[1], d] = True

    def set_boundaries(self, b):
        self.fixed = b

    def __call__(self):
        '''
        It returns the 1D arrays directly usable by a Topopt2D object
        '''
        ndof = 2*self.shape[0]*self.shape[1]

        f = np.zeros((ndof, 1))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (self.forces[i, j, 0] != 0) or (self.forces[i, j, 1] != 0):
                    idx = i*self.shape[1] + j
                    f[2*idx, 0] = self.forces[i, j, 0]
                    f[2*idx + 1, 0] = self.forces[i, j, 1]

        tabx, taby = np.where(self.fixed[..., 0].astype(np.uint8) == 1)
        indices = taby + self.shape[1]*tabx
        fixed_list = [2*indices]

        tabx, taby = np.where(self.fixed[..., 1].astype(np.uint8) == 1)
        indices = taby + self.shape[1]*tabx
        fixed_list.append(2*indices + 1)

        return f, np.concatenate(fixed_list)
