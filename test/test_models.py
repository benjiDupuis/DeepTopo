from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.training.trainer import DeepTopo
from deeptopo.topoptim.loadcase import MichellStructure


def test_gaussian_embedding():

    gauss = GaussianEmbedding(10)
    emb = gauss((5, 5))
    assert emb.shape == (25, 10), "wrong result shape"


def test_fcnn():

    gauss = GaussianEmbedding(10)
    net = FCNN([10, 10], 10)
    res = net(gauss((5, 5)))

    assert res.shape == (25, 1), "output of the network has wrong shape"


def test_deeptopo():

    shape = (5, 5)

    _ = DeepTopo(MichellStructure(shape),
                 GaussianEmbedding(10),
                 FCNN([10, 10]),
                 0.4)
