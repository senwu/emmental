from sklearn.metrics import roc_auc_score

from emmental.utils.utils import pred_to_prob


def roc_auc_scorer(gold, probs, preds, pos_label=1):
    """ROC AUC.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Recall.
    :rtype: dict
    """

    gold_probs = pred_to_prob(gold, n_classes=probs.shape[1])
    roc_auc = roc_auc_score(gold_probs, probs)

    return {"roc_auc": roc_auc}
