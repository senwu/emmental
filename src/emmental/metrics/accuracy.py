"""Emmental accuracy scorer."""
from typing import Dict, List, Optional, Union

import numpy as np
from numpy import ndarray

from emmental.utils.utils import prob_to_pred


def accuracy_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
    normalize: bool = True,
    topk: int = 1,
) -> Dict[str, Union[float, int]]:
    """Accuracy classification score.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      normalize: Normalize the results or not, defaults to True.
      topk: Top K accuracy, defaults to 1.

    Returns:
      Accuracy, if normalize is True, return the fraction of correctly predicted
      samples (float), else returns the number of correctly predicted samples (int).
    """
    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    if topk == 1 and preds is not None:
        n_matches = np.where(golds == preds)[0].shape[0]
    else:
        topk_preds = probs.argsort(axis=1)[:, -topk:][:, ::-1]
        n_matches = np.logical_or.reduce(
            topk_preds == golds.reshape(-1, 1), axis=1
        ).sum()

    if normalize:
        return {
            "accuracy" if topk == 1 else f"accuracy@{topk}": n_matches / golds.shape[0]
        }
    else:
        return {"accuracy" if topk == 1 else f"accuracy@{topk}": n_matches}
