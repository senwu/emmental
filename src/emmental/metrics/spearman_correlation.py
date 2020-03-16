from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray
from scipy.stats import spearmanr


def spearman_correlation_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
    return_pvalue: bool = False,
) -> Dict[str, float]:
    r"""Spearman rank-order correlation coefficient and the p-value.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray): Predicted probabilities.
      preds(ndarray or None): Predicted values.
      uids(list, optional): Unique ids, defaults to None.
      return_pvalue(bool, optional): Whether return pvalue or not, defaults to False.

    Returns:
      dict: Spearman rank-order correlation coefficient (with pvalue if return_pvalue
      is True).

    """

    probs = np.vstack(probs).squeeze()
    correlation, pvalue = spearmanr(golds, probs)

    if return_pvalue:
        return {"spearman_correlation": correlation, "spearman_pvalue": pvalue}

    return {"spearman_correlation": correlation}
