import numpy as np


def recall_scorer(gold, probs, preds, pos_label=1):
    """Recall.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Recall.
    :rtype: dict
    """

    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(gold == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FN = np.sum(np.logical_not(pred_pos) * gt_pos)

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return {"recall": recall}
