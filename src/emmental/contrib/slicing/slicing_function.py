import logging
from functools import wraps
from types import SimpleNamespace
from typing import Callable, List

import numpy as np
import torch
from numpy import ndarray

from emmental.data import EmmentalDataset

logger = logging.getLogger(__name__)


class slicing_function:
    r"""When wrapped with this decorator, slicing functions only need to return an
      indicator for whether an individual example (bundle of attributes) belongs
      in that slice. Iterating through the dataset, making the pred array (and
      masking), etc. are all handled automatically.

    Args:
      fields(list): Data attributes to use, defaults to [].

    """

    def __init__(self, fields: List[str] = []) -> None:
        self.fields = fields

    def __call__(self, f: Callable) -> Callable:
        @wraps(f)
        def wrapped_f(dataset: EmmentalDataset) -> ndarray:
            """

            Args:
              dataset(EmmentalDataset): Dataset to apply slicing function.

            Returns:
              ndarray: Indicators.

            """
            inds = []
            for idx in range(len(dataset)):
                example = SimpleNamespace(
                    **{field: dataset.X_dict[field][idx] for field in self.fields}
                )
                in_slice = f(example)
                inds.append(1 if in_slice else 0)
            inds = torch.from_numpy(np.array(inds)).view(-1)  # type: ignore
            logger.info(
                f"Total {int((inds == 1).sum())} / {len(dataset)} "  # type: ignore
                f" examples are "
                f"in slice {f.__name__}"
            )
            return inds

        return wrapped_f
