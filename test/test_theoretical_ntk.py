import os

from deeptopo.ntk.theoretical_ntk import TheoreticalNTK
from deeptopo.models.networks import FCNN
from deeptopo.models.embeddings import GaussianEmbedding, TorusEmbedding
from deeptopo.models.activations import NormalizedReLu


def test_th_ntk_gaussian():

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    net = FCNN(activation=NormalizedReLu(), beta=0.9)
    embedding = GaussianEmbedding(1, 10.)
    ntk = TheoreticalNTK(net, embedding)

    one_line_ntk = ntk.one_line_ntk((5, 5))
    assert one_line_ntk.shape == (5, 5), "one line ntk has wrong shape"


def test_th_ntk_torus():

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    net = FCNN(activation=NormalizedReLu(), beta=0.9)
    embedding = TorusEmbedding()
    ntk = TheoreticalNTK(net, embedding)

    one_line_ntk = ntk.one_line_ntk((5, 5))
    assert one_line_ntk.shape == (5, 5), "one line ntk has wrong shape"
