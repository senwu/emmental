from sklearn.metrics import roc_auc_score

from emmental.utils.utils import pred_to_prob


def roc_auc_scorer(golds, probs, preds, uids=None, pos_label=1):
    """ROC AUC.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: k-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :return: Recall.
    :rtype: dict
    """

    gold_probs = pred_to_prob(golds, n_classes=probs.shape[1])
    roc_auc = roc_auc_score(gold_probs, probs)

    return {"roc_auc": roc_auc}
