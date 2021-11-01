import torch

from deeptopo.ntk.empirical_ntk import empirical_ntk_1D
from deeptopo.models.networks import FCNN


def test_empirical_ntk():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_input = torch.rand(3, 2, device=device, dtype=torch.float)
    net = FCNN([10], 2)
    net.to(device)
    ntk = empirical_ntk_1D(batch_input, net)

    assert len(ntk.shape) == 2, "resulting ntk should be of dim 2"
    assert ntk.shape[0] == 3, "dim 0 of the ntk is wrong"
    assert ntk.shape[1] == 3, "dim 1 of the ntk is wrong"
