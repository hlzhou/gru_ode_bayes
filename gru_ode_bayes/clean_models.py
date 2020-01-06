"""GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps"""

import torch
import math
import numpy as np
from torchdiffeq import odeint

from torch.nn.utils.rnn import pack_padded_sequence


class GRUODECell(torch.nn.Module):
  def __init__(self, input_size, hidden_size, bias=True):
    """For p(t) modelling input_size should be 2x the x size."""
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
    self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

    self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, x, h):
    """
    Returns a change due to one step of using GRU-ODE for all h.

    Args:
        x        input values
        h        hidden state (current)
        delta_t  time step

    Returns:
        Updated h
    """
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
    n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

    dh = (1 - z) * (n - h)
    return dh


class GRUODECell_Autonomous(torch.nn.Module):
  def __init__(self, hidden_size, bias=True):
    """
    For p(t) modelling input_size should be 2x the x size.
    """
    super().__init__()
    self.hidden_size = hidden_size
    self.bias = bias

    # self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
    # self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

    self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, t, h):
    """
    Returns a change due to one step of using GRU-ODE for all h.
    The step size is given by delta_t.

    Args:
        t        time
        h        hidden state (current)

    Returns:
        Updated h
    """
    x = torch.zeros_like(h)
    z = torch.sigmoid(x + self.lin_hz(h))
    n = torch.tanh(x + self.lin_hn(z * h))

    dh = (1 - z) * (n - h)
    return dh