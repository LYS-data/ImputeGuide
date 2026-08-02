import numpy as np
import torch


def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "prelu":
        return torch.nn.PReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif (activation is None) or (activation == "none"):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
