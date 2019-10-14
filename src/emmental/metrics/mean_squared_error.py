from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean squared error regression loss.

    :param golds: Ground truth (correct) target values.
    :type golds: k-d np.array
    :param probs: Predicted target probabilities.
    :type probs: k-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array or None
    :param uids: Unique ids.
    :type uids: list, optional
    :return: Mean squared error regression loss.
    :rtype: dict
    """

    return {"mean_squared_error": float(mean_squared_error(golds, probs))}
