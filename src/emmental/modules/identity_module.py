"""Emmental identity module."""
from torch import Tensor, nn as nn


class IdentityModule(nn.Module):
    """An identity module that outputs the input."""

    def __init__(self) -> None:
        """Initialize IdentityModule."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Forward function.

        Args:
          x: Input tensor.

        Returns:
          Output of identity module which is the same with input.
        """
        return input
