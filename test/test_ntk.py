import torch

from deeptopo.ntk.empirical_ntk import empirical_ntk, empirical_NTK_one_line
from deeptopo.models.networks import FCNN


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_empirical_ntk():

    device = torch.device(DEVICE)

    batch_input = torch.rand(3, 2, device=device, dtype=torch.float)
    net = FCNN([10], 2)
    net.to(device)
    ntk = empirical_ntk(batch_input, net)

    assert len(ntk.shape) == 2, "resulting ntk should be of dim 2"
    assert ntk.shape[0] == 3, "dim 0 of the ntk is wrong"
    assert ntk.shape[1] == 3, "dim 1 of the ntk is wrong"


def test_empirical_one_line_ntk():

    device = torch.device(DEVICE)

    batch_input = torch.rand(3, 2, device=device, dtype=torch.float)
    net = FCNN([10], 2)
    net.to(device)
    one_line_ntk = empirical_NTK_one_line(batch_input, net, 0)

    assert len(one_line_ntk.shape) == 2, "resulting ntk should be of dim 2"
    assert one_line_ntk.shape[0] == 1, "dim 0 of the ntk is wrong"
    assert one_line_ntk.shape[1] == 3, "dim 1 of the ntk is wrong"
