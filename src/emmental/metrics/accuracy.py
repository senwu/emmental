from typing import Dict, List, Optional, Union

import numpy as np
from numpy import ndarray


def accuracy_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    normalize: bool = True,
) -> Dict[str, Union[float, int]]:
    r"""Accuracy classification score.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray or None): Predicted probabilities.
      preds(ndarray): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      normalize(bool, optional): Normalize the results or not, defaults to True.

    Returns:
      dict: Accuracy, if normalize is True, return the fraction of correctly
        predicted samples (float), else returns the number of correctly predicted
        samples (int).

    """

    if normalize:
        return {"accuracy": np.where(golds == preds)[0].shape[0] / golds.shape[0]}
    else:
        return {"accuracy": np.where(golds == preds)[0].shape[0]}
