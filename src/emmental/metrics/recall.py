"""Emmental recall scorer."""
from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from emmental.utils.utils import prob_to_pred


def recall_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Recall.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      pos_label: The positive class label, defaults to 1.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.

    Returns:
      Recall.
    """
    assert sample_scores is None
    assert return_sample_scores is False

    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(golds == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FN = np.sum(np.logical_not(pred_pos) * gt_pos)

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return {"recall": recall}
