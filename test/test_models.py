from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN


def test_gaussian_embedding():

    gauss = GaussianEmbedding(10)
    emb = gauss((5, 5))
    assert emb.shape == (25, 10), "wrong result shape"


def test_fcnn():

    gauss = GaussianEmbedding(10)
    net = FCNN(gauss, [10, 10])
    res = net((5, 5))

    assert res.shape == (25, 1), "output of the network has wrong shape"
