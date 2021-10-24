from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN


def test_gaussian_embedding():

    gauss = GaussianEmbedding((5, 5), 10)
    assert gauss.grid.shape == (25, 2), "wrong embedding shape"
    emb = gauss()
    assert emb.shape == (25, 10), "wrong result shape"


def test_fcnn():

    gauss = GaussianEmbedding((5, 5), 10)
    net = FCNN(gauss, [10, 10])
    res = net()

    assert res.shape == (25, 1), "output of the network has wrong shape"
