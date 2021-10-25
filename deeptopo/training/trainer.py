import numpy as np
import torch
from tqdm import tqdm

from deeptopo.topoptim.topopt2D import Topopt2D
from deeptopo.topoptim.loadcase import LoadCase
from deeptopo.models.networks import FCNN
from deeptopo.training.training_utils import root_finder


class DeepTopo(Topopt2D):

    lr: float = 1.e-3
    gamma: float = 1.e-2

    def __init__(self, loadcase: LoadCase,
                 net: FCNN,
                 volfrac: float):

        super(DeepTopo, self).__init__(loadcase, volfrac, method=None, ft=2)
        self.net = net
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net.to(self.device)

    def __call__(self, iter: int, display: bool = True, initial_field: bool = True):

        optimizer = torch.optim.Rprop(self.net.parameters(), lr=self.lr)
        target_volume = self.volfrac*self.shape[0]*self.shape[1]

        # Substracting the initial density field to simulate a constant initialization
        with torch.no_grad():
            field_0 = self.net(self.shape)
            bias_0 = np.log(self.volfrac/(1. - self.volfrac))

        for k in tqdm(range(iter + 1)):

            # if initial field is True, forward and substraction of initial density field
            if initial_field:
                xhat = self.net(self.shape) - field_0 + bias_0
            else:
                xhat = self.net(self.shape)

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
