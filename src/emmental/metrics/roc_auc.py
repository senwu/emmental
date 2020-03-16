import logging
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import roc_auc_score

from emmental.utils.utils import pred_to_prob, prob_to_pred

logger = logging.getLogger(__name__)


def roc_auc_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """ROC AUC.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray): Predicted probabilities.
      preds(ndarray or None): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      pos_label(int, optional): The positive class label, defaults to 1.

    Returns:
      dict: ROC AUC score.

    """

    if len(probs.shape) == 2 and probs.shape[1] == 1:
        probs = probs.reshape(probs.shape[0])

    if len(golds.shape) == 2 and golds.shape[1] == 1:
        golds = golds.reshape(golds.shape[0])

    if len(probs.shape) > 1:
        if len(golds.shape) > 1:
            golds = pred_to_prob(prob_to_pred(golds), n_classes=probs.shape[1])
        else:
            golds = pred_to_prob(golds, n_classes=probs.shape[1])
    else:
        if len(golds.shape) > 1:
            golds = prob_to_pred(golds)

    try:
        roc_auc = roc_auc_score(golds, probs)
    except ValueError:
        logger.warning(
            "Only one class present in golds."
            "ROC AUC score is not defined in that case, set as nan instead."
        )
        roc_auc = float("nan")

    return {"roc_auc": roc_auc}
