import numpy as np


def precision_scorer(gold, probs, preds, pos_label=1):
    pred_pos = np.where(preds == pos_label, True, False)
    gt_pos = np.where(gold == pos_label, True, False)
    TP = np.sum(pred_pos * gt_pos)
    FP = np.sum(pred_pos * np.logical_not(gt_pos))

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0

    return {"precision": precision}
