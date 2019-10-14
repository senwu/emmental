import logging
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import roc_auc_score

from emmental.utils.utils import pred_to_prob

logger = logging.getLogger(__name__)


def roc_auc_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """ROC AUC.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: k-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array or None
    :param uids: Unique ids.
    :type uids: list, optional
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
