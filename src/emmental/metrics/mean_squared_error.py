"""Emmental mean squared error scorer."""
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean squared error regression loss.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.

    Returns:
      Mean squared error regression loss.
    """
    return {"mean_squared_error": float(mean_squared_error(golds, probs))}
