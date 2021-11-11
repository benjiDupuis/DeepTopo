from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from deeptopo.models.embeddings import Embedding2D


def empirical_ntk(batch_input: torch.Tensor, net: nn.Module) -> torch.Tensor:
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

    with torch.no_grad():
        gradient_tensor = torch.stack(gradient_list)

        return torch.einsum('ij, jk->ik', gradient_tensor, gradient_tensor.T)


def empirical_NTK_one_line(batch_input: torch.Tensor, net: nn.Module, index: int)\
        -> torch.Tensor:
    """
    Computation of one line of the empirical NTK \n
    :param batch_input: batch of point to evaluate the NTK, shape (B,I) \n
    :net: nn.Module to be differentiated \n
    :index: data point wrt which the line is computed \n
    """

    assert isinstance(index, int), "index must be an int"
    assert len(batch_input.shape) == 2, "in this function, batch is of dim 2"
    assert index < batch_input.shape[0] and index >= 0, \
        "index must correspond to n element of the batch"

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

    with torch.no_grad():

        gradient_tensor = torch.stack(gradient_list)

        return torch.einsum('ij, jk->ik', gradient_tensor[index].unsqueeze(0),
                            gradient_tensor.T)


def one_line_ntk_with_embedding(net: nn.Module,
                                shape: Tuple[int],
                                embedding: Optional[Embedding2D] = None,
                                index: Optional[int] = None):
    """
    Compute one line of the NTK of embedding + net 
    with respect to index
    """
    if embedding is None:
        batch_input = Embedding2D.make_grid(shape)
    else:
        batch_input = embedding(shape)

    index = shape[1]*(1 + shape[0])//2 if index is None else index

    ntk = empirical_NTK_one_line(batch_input, net, index)
    ntk = ntk[0].reshape(shape[0], shape[1]).cpu().detach().data.numpy()

    return ntk
