"""Emmental accuracy scorer."""
from typing import Dict, List, Optional, Union

import numpy as np
from numpy import ndarray

from emmental.utils.utils import prob_to_pred


def accuracy_scorer(
    golds: Optional[ndarray] = None,
    probs: Optional[ndarray] = None,
    preds: Optional[ndarray] = None,
    uids: Optional[List[str]] = None,
    sample_scores: Optional[Dict[str, Union[float, int]]] = None,
    return_sample_scores: bool = False,
    normalize: bool = True,
    topk: int = 1,
) -> Dict[str, Union[float, int]]:
    """Accuracy classification score.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample instead of overall
        score, default to False.
      normalize: Normalize the results or not, defaults to True.
      topk: Top K accuracy, defaults to 1.

    Returns:
      Accuracy, if normalize is True, return the fraction of correctly predicted
      samples (float), else returns the number of correctly predicted samples (int).
    """
    metric_name = "accuracy" if topk == 1 else f"accuracy@{topk}"

    # Calculate accuracy score using sample scores.
    if sample_scores is not None:
        assert metric_name in sample_scores
        scores = np.array(sample_scores[metric_name]).reshape(-1)
        return {
            metric_name: float(scores.sum() / (scores.shape[0] if normalize else 1))
        }

    # Calculate accuracy score using probs, preds, and golds.
    # Convert probabilistic label to hard label
    if golds is not None and len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    if topk == 1 and preds is not None:
        sample_score = (golds == preds).astype(int)
    else:
        topk_preds = probs.argsort(axis=1)[:, -topk:][:, ::-1]
        sample_score = np.logical_or.reduce(
            topk_preds == golds.reshape(-1, 1), axis=1
        ).astype(int)

    # Return sample scores
    if return_sample_scores:
        return {metric_name: sample_score}

    # Return overall score
    return {
        metric_name: float(
            sample_score.sum() / (sample_score.shape[0] if normalize else 1)
        )
    }
