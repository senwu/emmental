import numpy as np


def recall_scorer(gold, preds, probs, pos_label=1):
    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(gold == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FN = np.sum(np.logical_not(pred_pos) * gt_pos)

    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return {"recall": recall}
