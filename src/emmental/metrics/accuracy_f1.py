"""Emmental accuracy f1 scorer."""
from statistics import mean
from typing import Dict, List, Optional

from numpy import ndarray

from emmental.metrics.accuracy import accuracy_scorer
from emmental.metrics.fbeta import f1_scorer


def accuracy_f1_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Average of accuracy and f1 score.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      pos_label: The positive class label, defaults to 1.

    Returns:
      Average of accuracy and f1.
    """
    metrics = dict()
    accuracy = accuracy_scorer(golds, probs, preds, uids)
    f1 = f1_scorer(golds, probs, preds, uids, pos_label=pos_label)
    metrics["accuracy_f1"] = mean([accuracy["accuracy"], f1["f1"]])

    return metrics
