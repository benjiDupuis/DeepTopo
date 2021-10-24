from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn

from deeptopo.models.embeddings import Embedding


# Bias term of the Linear layers in the NTK parameterization
class BiasLayer(nn.Module):

    def __init__(self, n: int, beta: float):
        super(BiasLayer, self).__init__()
        self.size = n
        self.beta = beta
        self.weight = nn.Parameter(torch.Tensor(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta*self.weight


class FCNN:

    def __init__(self, embedding: Embedding, inter_layers: List[int],
                 beta: float = 0.2, omega: float = 1., activation=torch.sigmoid):
        '''
        inter_layers: intermediary layers size
        (input is embedding size and output is 1)
        '''
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.beta = beta
        self.omega = omega
        self.embedding = embedding
        self.layers = [self.embedding.size] + inter_layers + [1]
        self.L = len(self.layers) - 1
        self.bias_modules = nn.ModuleList()
        self.lin_modules = nn.ModuleList()
        self.act = activation

        for k in range(0, self.L - 1):
            self.lin_modules.append(nn.Linear(self.layers[k], self.layers[k+1], bias=False))
            self.bias_modules.append(BiasLayer(self.layers[k+1], self.beta))
        self.lin_modules.append(nn.Linear(self.layers[self.L - 1], self.layers[self.L], bias=False))
        self.bias_modules.append(BiasLayer(self.layers[self.L], self.beta))

        self.__init_weights()

    def __call__(self):
        x = self.embedding().type(torch.FloatTensor)
        return self.forward(x)

    def forward(self, x):

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
