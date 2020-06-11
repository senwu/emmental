"""Slicing modules."""
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
      module_name: Module name.
      intermediate_output_dict: output dict.
      Y: Gold lables.
      active: Active sample index.
      weight: Class weights.

    Returns:
      Loss.
    """
    return F.cross_entropy(
        intermediate_output_dict[module_name][0][active],
        (Y.view(-1) - 1)[active],
        weight,
    )


def output(module_name: str, intermediate_output_dict: Dict[str, Any]) -> Tensor:
    """Output function.

    Args:
      module_name: Module name.
      intermediate_output_dict: output dict.

    Returns:
      Output.
    """
    return F.softmax(intermediate_output_dict[module_name][0], dim=1)
