"""Emmental spearman correlation scorer."""
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
    """Spearman rank-order correlation coefficient and the p-value.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      return_pvalue: Whether return pvalue or not, defaults to False.

    Returns:
      Spearman rank-order correlation coefficient with pvalue if return_pvalue is True.
    """
    probs = np.vstack(probs).squeeze()  # type: ignore
    correlation, pvalue = spearmanr(golds, probs)

    if return_pvalue:
        return {"spearman_correlation": correlation, "spearman_pvalue": pvalue}

    return {"spearman_correlation": correlation}
