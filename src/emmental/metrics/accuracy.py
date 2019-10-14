from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray


def accuracy_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """Accuracy classification score.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array or None
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list, optional
    :param normalize: Normalize the results or not, defaults to True
    :param normalize: bool, optional
    :return: Accuracy, if normalize == True, return the fraction of correctly
        predicted samples (float), else returns the number of correctly predicted
        samples (int).
    :rtype: dict
    """

    if normalize:
        return {"accuracy": np.where(golds == preds)[0].shape[0] / golds.shape[0]}
    else:
        return {"accuracy": np.where(golds == preds)[0].shape[0]}
