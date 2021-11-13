import fire

from deeptopo.models.activations import NormalizedReLu
from deeptopo.ntk.theoretical_ntk import TheoreticalNTK
from deeptopo.models.networks import FCNN
from deeptopo.models.embeddings import GaussianEmbedding

OUTPUT_DIR = "results"


def main():

    net = FCNN(activation=NormalizedReLu(), beta=0.9)  # by default only one interemediate layer
    embedding = GaussianEmbedding(1, 10.)

    ntk = TheoreticalNTK(net, embedding)

    ntk.visualize_one_line_ntk((40, 40), OUTPUT_DIR)


if __name__ == "__main__":
    fire.Fire(main)
