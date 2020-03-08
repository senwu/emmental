from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from emmental.utils.utils import prob_to_pred


def recall_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Recall.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray or None): Predicted probabilities.
      preds(ndarray): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      pos_label(int, optional): The positive class label, defaults to 1.

    Returns:
      dict: Recall.

    """

    # Convert probabilistic label to hard label
    if len(golds.shape) == 2:
        golds = prob_to_pred(golds)

    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(golds == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FN = np.sum(np.logical_not(pred_pos) * gt_pos)

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return {"recall": recall}
