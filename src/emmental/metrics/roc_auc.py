import logging

from sklearn.metrics import roc_auc_score

from emmental.utils.utils import pred_to_prob

logger = logging.getLogger(__name__)


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

    try:
        roc_auc = roc_auc_score(gold_probs, probs)
    except ValueError:
        logger.warning(
            "Only one class present in golds."
            "ROC AUC score is not defined in that case, set as nan instead."
        )
        roc_auc = float("nan")

    return {"roc_auc": roc_auc}
