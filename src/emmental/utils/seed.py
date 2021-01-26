# Copyright (c) 2021 Sen Wu. All Rights Reserved.


"""Helper function to set random seed for reproducibility of models."""

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seed for random, numpy, and pytorch.

    Args:
      seed: The random seed, defaults to `None` which select it randomly.
    """
    max_value = np.iinfo(np.uint32).max
    min_value = np.iinfo(np.uint32).min

    try:
        seed = int(seed)
        logger.info(f"Set random seed to {seed}.")
    except (TypeError, ValueError):
        seed = random.randint(min_value, max_value)
        logger.info(f"No random seed specified, randomly set random seed to {seed}.")

    if not (min_value <= seed <= max_value):
        new_seed = random.randint(min_value, max_value)
        logger.info(
            f"Random seed {seed} is not valid, randomly set random seed to {new_seed}."
        )
        seed = new_seed

    # Set random seed for random
    random.seed(seed)
    # Set random seed for all numpy operations
    np.random.seed(seed=seed)
    # Set random seed for PyTorch
    torch.manual_seed(seed)
