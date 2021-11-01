import torch
import torch.nn as nn
from tqdm import tqdm


def empirical_ntk_1D(batch_input: torch.Tensor, net: nn.Module) -> torch.Tensor:
    """
    :param input_batch: batch of shape (B, I) where B is the 
    number of inputs points and I is the input shape to the network \n
    :param net: nn.Module, output shape of net is supposed to be 1 here \n
    :output: empirical NTK Gram matrix of shape I x I
    """
    assert len(batch_input.shape) == 2, "in this function, batch is of dim 2"

    batch_output = net(batch_input)
    assert len(batch_output.shape) == 2, "output should be of dim 2"
    assert batch_output.shape[1] == 1, "output of the network should be a scalar in this function"

    # We compute each gradient
    gradient_list = []
    for b in tqdm(range(batch_input.shape[0])):
        net.zero_grad()
        batch_output[b].backward(retain_graph=True)
        gradient = torch.cat([p.grad.flatten() for p in net.parameters()])
        gradient_list.append(gradient)

    gradient_list = torch.stack(gradient_list)

    return torch.einsum('ij, jk->ik', gradient_list, gradient_list.T)
