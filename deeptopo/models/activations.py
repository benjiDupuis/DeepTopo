import numpy as np
import torch

# Some normalized activation functions
def normalized_relu(x):
    return np.sqrt(2.)*torch.relu(x)

def norm_sin(a):
    return np.sqrt(2./(1 - np.exp(-2.*a*a)))

def norm_cos(a):
    return np.sqrt(2./(1 + np.exp(-2.*a*a)))

def norm_exp(a):
    return np.exp(a**2)
