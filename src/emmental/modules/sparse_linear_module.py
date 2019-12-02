import math

import torch
import torch.nn as nn
from torch import Tensor


class SparseLinear(nn.Module):
    r"""A sparse linear module.

    Args:
      num_features(int): Size of features.
      num_classes(int): Number of classes.
      bias(bool): Use bias term or not, defaults to False.
      padding_idx(int): padding index, defaults to 0.

    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        bias: bool = False,
        padding_idx: int = 0,
    ) -> None:

        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        self.weight = nn.Embedding(
            self.num_features, self.num_classes, padding_idx=self.padding_idx
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_classes))  # type: ignore
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reinitiate the weight parameters."""

        stdv = 1.0 / math.sqrt(self.num_features)
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.weight.weight.data[self.padding_idx].fill_(0)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:  # type: ignore
        r"""Forward function.

        Args:
          x(Tensor): Feature indices.
          w(Tensor): Feature weights.

        Returns:
          Tensor: Output of linear layer.

        """

        if self.bias is None:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1)
        else:
            return (w.unsqueeze(2) * self.weight(x)).sum(dim=1) + self.bias
