import torch.nn as nn
from torch import Tensor


class IdentityModule(nn.Module):
    r"""An identity module that outputs the input."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        r"""Forward function.

        Args:
          x(Tensor): Input tensor.

        Returns:
          Tensor: Output of identity module which is the same with input.

        """

        return input
