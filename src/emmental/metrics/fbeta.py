from typing import Dict, List, Optional

from numpy import ndarray

from emmental.metrics.precision import precision_scorer
from emmental.metrics.recall import recall_scorer
from emmental.utils.utils import prob_to_pred


def fbeta_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    pos_label: int = 1,
    beta: int = 1,
) -> Dict[str, float]:
    """F-beta score is the weighted harmonic mean of precision and recall.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray or None): Predicted probabilities.
      preds(ndarray): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      pos_label(int, optional): The positive class label, defaults to 1.
      beta(float, optional): Weight of precision in harmonic mean, defaults to 1.

    Returns:
      dict: F-beta score.

    """

    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    precision = precision_scorer(golds, probs, preds, uids, pos_label)["precision"]
    recall = recall_scorer(golds, probs, preds, uids, pos_label)["recall"]

    fbeta = (
        (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        if (beta ** 2 * precision) + recall > 0
        else 0.0
    )

    return {f"f{beta}": fbeta}


def f1_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """F-1 score.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray or None): Predicted probabilities.
      preds(ndarray): Predicted values.
      uids(list, optional): Unique ids.
      pos_label(int, optional): The positive class label, defaults to 1.

    Returns:
      dict: F-1 score.

    """

    return {"f1": fbeta_scorer(golds, probs, preds, uids, pos_label, beta=1)["f1"]}
