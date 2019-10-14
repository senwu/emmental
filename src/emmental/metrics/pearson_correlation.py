from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr


def pearson_correlation_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
    return_pvalue: bool = False,
) -> Dict[str, float]:
    """Pearson correlation coefficient and the p-value.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array or None
    :param uids: Unique ids.
    :type uids: list, optional
    :para return_pvalue: Whether return pvalue.
    :type return_pvalue: bool, optional
    :return: Pearson correlation coefficient and the p-value.
    :rtype: dict
    """

    probs = np.vstack(probs).squeeze()
    correlation, pvalue = pearsonr(golds, probs)
    if np.isnan(correlation):
        correlation = 0.0
        pvalue = 0.0

    if return_pvalue:
        return {"pearson_correlation": correlation, "pearson_pvalue": pvalue}

    return {"pearson_correlation": correlation}
