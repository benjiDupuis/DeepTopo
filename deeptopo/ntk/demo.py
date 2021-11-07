import fire
import torch

from deeptopo.models.embeddings import GaussianEmbedding
from deeptopo.models.networks import FCNN
from deeptopo.ntk.ntk_visualization import visualize_ntk
from deeptopo.models.activations import NormalizedSin


def main(output_dir: str = "results"):
    """
    example of visualization of the empirical NTK
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    shape = (40, 40)
    embedding = GaussianEmbedding(100, ell=10.)
    network = FCNN([100], input_size=100, beta=0.9, activation=NormalizedSin(1.))
    network.to(device)

    visualize_ntk(network, shape, output_dir, embedding)


if __name__ == "__main__":
    fire.Fire(main)
