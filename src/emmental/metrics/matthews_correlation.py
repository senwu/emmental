from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Matthews correlation coefficient (MCC).

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array or None
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list, optional
    :return: Matthews correlation coefficient score.
    :rtype: dict
    """

    return {"matthews_corrcoef": matthews_corrcoef(golds, preds)}
