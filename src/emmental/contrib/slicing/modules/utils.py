from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor


def ce_loss(
    module_name: str,
    intermediate_output_dict: Dict[str, Any],
    Y: Tensor,
    active: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Cross entropy loss.

    Args:
      module_name(str): Module name.
      intermediate_output_dict(dict): output dict.
      Y(Tensor): Gold lables.
      active(Tensor): Active sample index.
      weight(Tensor, optional): Class weights.

    Returns:
      Tensor: Loss.

    """

    return F.cross_entropy(
        intermediate_output_dict[module_name][0][active],
        (Y.view(-1) - 1)[active],
        weight,
    )


def output(module_name: str, intermediate_output_dict: Dict[str, Any]) -> Tensor:
    """Output function.

    Args:
      module_name(str): Module name.
      intermediate_output_dict(dict): output dict.

    Returns:
      Tensor: Output.
    """

    return F.softmax(intermediate_output_dict[module_name][0], dim=1)
