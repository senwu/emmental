import torch.nn as nn


class IdentityModule(nn.Module):
    """An identity module that outputs the input."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward function.
        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output of identity module which is the same with input.
        :rtype: torch.Tensor
        """

        return x
