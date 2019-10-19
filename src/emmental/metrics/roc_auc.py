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

    Args:
      golds(np.array): Ground truth values.
      probs(np.array): Predicted probabilities.
      preds(np.array or None): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      pos_label(int, optional): The positive class label, defaults to 1.

    Returns:
      dict: ROC AUC score.

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
