"""Emmental mean squared error scorer."""
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
) -> Dict[str, float]:
    """Mean squared error regression loss.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.

    Returns:
      Mean squared error regression loss.
    """
    assert sample_scores is None
    assert return_sample_scores is False

    return {"mean_squared_error": float(mean_squared_error(golds, probs))}
