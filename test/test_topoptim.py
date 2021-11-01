from deeptopo.topoptim.topopt2D import Topopt2D
from deeptopo.topoptim.loadcase_zoo import MichellStructure


def test_topopt2D():

    loadcase = MichellStructure((100, 50))
    optim = Topopt2D(loadcase, 0.4)
    assert optim.shape == (100, 50), "wrong shape in optimizer"
