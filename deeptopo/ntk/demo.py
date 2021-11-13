import fire
import torch

from deeptopo.models.activations import NormalizedReLu
from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.ntk.ntk_visualization import visualize_ntk


def main(output_dir: str = "results"):
    """
    example of visualization of the empirical NTK
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    shape = (40, 40)
    embedding = GaussianEmbedding(100, ell=10.)
    network = FCNN([1000], input_size=100, beta=0.9, activation=NormalizedReLu())
    network.to(device)

    visualize_ntk(network, shape, output_dir, embedding)


if __name__ == "__main__":
    fire.Fire(main)
