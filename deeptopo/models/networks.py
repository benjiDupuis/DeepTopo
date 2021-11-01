from typing import List

import numpy as np
import torch
import torch.nn as nn


# Bias term of the Linear layers in the NTK parameterization
class BiasLayer(nn.Module):

    def __init__(self, n: int, beta: float):
        super(BiasLayer, self).__init__()
        self.size = n
        self.beta = beta
        self.weight = nn.Parameter(torch.Tensor(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta*self.weight


class FCNN(nn.Module):

    def __init__(self, inter_layers: List[int], input_size: int = 4,
                 beta: float = 0.2, activation=torch.relu):
        '''
        inter_layers: intermediary layers size
        (input is embedding size and output is 1)
        '''
        super(FCNN, self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.beta = beta
        self.input_size = input_size
        self.inter_layers = inter_layers
        self.layers = [self.input_size] + inter_layers + [1]
        self.L = len(self.layers) - 1
        self.act = activation

        self.init_layers(self.input_size)

    def init_layers(self, size):
        """
        This method can be used to change input size
        in case of different embedding
        while preserving the remaining of the architecture
        """

        self.input_size = size
        self.layers = [self.input_size] + self.inter_layers + [1]

        self.bias_modules = nn.ModuleList()
        self.lin_modules = nn.ModuleList()

        for k in range(0, self.L - 1):
            self.lin_modules.append(nn.Linear(self.layers[k], self.layers[k+1], bias=False))
            self.bias_modules.append(BiasLayer(self.layers[k+1], self.beta))
        self.lin_modules.append(nn.Linear(self.layers[self.L - 1], self.layers[self.L], bias=False))
        self.bias_modules.append(BiasLayer(self.layers[self.L], self.beta))

        self.__init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)

        for k in range(self.L-1):
            x = np.sqrt((1. - self.beta**2)/self.layers[k])*self.lin_modules[k](x)
            x = self.bias_modules[k](x)
            x = self.act(x)

        x = np.sqrt((1. - self.beta**2)/self.layers[self.L - 1])*self.lin_modules[self.L - 1](x)
        x = self.bias_modules[-1](x)

        return x

    # Initialization in the NTK regime
    def __init_weights(self):
        for k in range(self.L - 1):
            self.lin_modules[k].weight.data.normal_(0., 1.)
            self.bias_modules[k].weight.data.normal_(0., 1.)
        self.lin_modules[-1].weight.data.normal_(0., 1.)
        self.bias_modules[-1].weight.data.normal_(0., 1.)
