"""Emmental matthews correlation coefficient scorer."""
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import matthews_corrcoef

from emmental.utils.utils import prob_to_pred


def matthews_correlation_coefficient_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Matthews correlation coefficient (MCC).

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.

    Returns:
      Matthews correlation coefficient score.
    """
    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    return {"matthews_corrcoef": matthews_corrcoef(golds, preds)}
