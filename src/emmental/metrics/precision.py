from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from emmental.utils.utils import prob_to_pred


def precision_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Precision.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array or None
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list, optional
    :param pos_label: The positive class label, defaults to 1
    :type pos_label: int, optional
    :return: Precision.
    :rtype: dict
    """
    if len(golds.shape) > 1:
        golds = prob_to_pred(golds)
    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(golds == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FP = np.sum(pred_pos * np.logical_not(gt_pos))

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0

    return {"precision": precision}
