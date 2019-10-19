from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Matthews correlation coefficient (MCC).

    Args:
      golds(np.array): Ground truth values.
      probs(np.array or None): Predicted probabilities.
      preds(np.array): Predicted values.
      uids(list, optional): Unique ids, defaults to None.

    Returns:
      dict: Matthews correlation coefficient score.

    """

    return {"matthews_corrcoef": matthews_corrcoef(golds, preds)}
