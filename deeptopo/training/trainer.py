import numpy as np
import torch
from tqdm import tqdm

from deeptopo.topoptim.topopt2D import Topopt2D
from deeptopo.topoptim.loadcase import LoadCase
from deeptopo.models.networks import FCNN
from deeptopo.training.training_utils import root_finder
from deeptopo.models.embeddings import Embedding2D


class DeepTopo(Topopt2D):

    lr: float = 1.e-3
    gamma: float = 1.e-2
    grid_size: float = 1.

    def __init__(self,
                 loadcase: LoadCase,
                 embedding: Embedding2D,
                 net: FCNN,
                 volfrac: float):

        super(DeepTopo, self).__init__(loadcase, volfrac, method=None, ft=2)
        self.embedding = embedding
        self.net = net

        # test if embedding size and input size in the net are the same
        # if not the case, the net is redefined
        if not(embedding.size == net.input_size):
            print("WARNING : embedding.size \
                and net.input_size are not the same,\
                network has been reinitialized")
            net.init_layers(embedding.size)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net.to(self.device)

    def __call__(self, iter: int, display: bool = True, initial_field: bool = True):

        optimizer = torch.optim.Rprop(self.net.parameters(), lr=self.lr)
        target_volume = self.volfrac*self.shape[0]*self.shape[1]
        embedded_input = self.embedding(self.shape, dr=self.grid_size)

        # Substracting the initial density field to simulate a constant initialization
        with torch.no_grad():
            field_0 = self.net(embedded_input)
            bias_0 = np.log(self.volfrac/(1. - self.volfrac))

        for k in tqdm(range(iter + 1)):

            # if initial field is True, forward and substraction of initial density field
            if initial_field:
                xhat = self.net(embedded_input) - field_0 + bias_0
            else:
                xhat = self.net(embedded_input)

            # Computation of the optimal bias
            with torch.no_grad():
                bias_opt = root_finder(xhat, target_volume, torch.sigmoid)

            # Computing derivative of the bias
            # Regularizing the optimal bias toward zero can improve regularity
            with torch.no_grad():
                x = torch.sigmoid(xhat + bias_opt)[:, 0]
                xdot = x*(1.-x) + 1.e-6
                db = -xdot/(xdot.sum().item())

            # Applying SIMP algorithm of x
            _, _ = self.step(0, x.cpu().detach().numpy())
            gradient = torch.tensor(self.gradient, device=self.device).type(torch.FloatTensor)

            # Gradient of C with respect to xhat, using implicit differenciation
            with torch.no_grad():
                dc_xhat = torch.mm(torch.einsum('i,j->ij', db, xdot.T) +
                                   torch.diag(xdot), gradient.unsqueeze(1))
                gr = dc_xhat + self.gamma*2*bias_opt*db.unsqueeze(1)

            # backward and optimization of the parameters of the network
            if k < iter:
                optimizer.zero_grad()
                xhat.backward(gr)
                optimizer.step()

            if display or (k == iter):
                print("Iteration ", k, "  Compliance : ", round(self.compliance, 2),
                      "  Biais Optimal : ", round(bias_opt, 2),
                      "  Volume : ", round(x.sum().item()/(self.shape[0]*self.shape[1]), 2),
                      "learning rate : ", self.lr)

        return x.clamp(0., 1.).cpu().detach().numpy().reshape(self.shape[0], self.shape[1]).T

    # @torch.no_grad()
    # def up_sampling(self, up_sampling_factor: float):

    #     finer_embedded_input = self.embedding(self.shape, up_sampling_factor*self.grid_size)

    #     up_sampled_field = self.net()

    #     # TODO: end this method
