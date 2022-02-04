"""Emmental fbeta scorer."""
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
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
    pos_label: int = 1,
    beta: int = 1,
) -> Dict[str, float]:
    """F-beta score is the weighted harmonic mean of precision and recall.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.
      pos_label: The positive class label, defaults to 1.
      beta: Weight of precision in harmonic mean, defaults to 1.

    Returns:
      F-beta score.
    """
    assert sample_scores is None
    assert return_sample_scores is False

    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    precision = precision_scorer(
        golds,
        probs,
        preds,
        uids,
        sample_scores,
        return_sample_scores=return_sample_scores,
        pos_label=pos_label,
    )["precision"]
    recall = recall_scorer(
        golds,
        probs,
        preds,
        uids,
        sample_scores,
        return_sample_scores=return_sample_scores,
        pos_label=pos_label,
    )["recall"]

    fbeta = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if (beta**2 * precision) + recall > 0
        else 0.0
    )

    return {f"f{beta}": fbeta}


def f1_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
    pos_label: int = 1,
) -> Dict[str, float]:
    """F-1 score.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.
      pos_label: The positive class label, defaults to 1.

    Returns:
      F-1 score.
    """
    return {
        "f1": fbeta_scorer(
            golds,
            probs,
            preds,
            uids,
            sample_scores,
            return_sample_scores=return_sample_scores,
            pos_label=pos_label,
            beta=1,
        )["f1"]
    }
