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
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
    pos_label: int = 1,
    topk: int = 1,
) -> Dict[str, float]:
    """Average of accuracy and f1 score.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.
      pos_label: The positive class label, defaults to 1.
      topk: Top K accuracy, defaults to 1.

    Returns:
      Average of accuracy and f1.
    """
    metrics = dict()
    accuracy = accuracy_scorer(
        golds,
        probs,
        preds,
        uids,
        sample_scores=sample_scores,
        return_sample_scores=return_sample_scores,
        topk=topk,
    )
    f1 = f1_scorer(
        golds,
        probs,
        preds,
        uids,
        sample_scores=sample_scores,
        return_sample_scores=return_sample_scores,
        pos_label=pos_label,
    )
    metrics["accuracy_f1" if topk == 1 else f"accuracy@{topk}_f1"] = mean(
        [accuracy["accuracy" if topk == 1 else f"accuracy@{topk}"], f1["f1"]]
    )

    return metrics
