import numpy as np

from emmental.utils.utils import prob_to_pred


def recall_scorer(golds, probs, preds, pos_label=1):
    """Recall.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Recall.
    :rtype: dict
    """
    if len(golds.shape) > 1:
        golds = prob_to_pred(golds)
    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(golds == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FN = np.sum(np.logical_not(pred_pos) * gt_pos)

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return {"recall": recall}
