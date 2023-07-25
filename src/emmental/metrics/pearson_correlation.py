"""Emmental pearson correlation scorer."""
from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr


def pearson_correlation_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
    sample_scores: Optional[Dict[str, float]] = None,
    return_sample_scores: bool = False,
    return_pvalue: bool = False,
) -> Dict[str, float]:
    """Pearson correlation coefficient and the p-value.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      sample_scores: Scores for each samples, defaults to None.
      return_sample_scores: Whether return score for each sample, default to False.
      return_pvalue: Whether return pvalue or not, defaults to False.

    Returns:
      Pearson correlation coefficient with pvalue if return_pvalue is True.
    """
    assert sample_scores is None
    assert return_sample_scores is False

    probs = np.vstack(probs).squeeze()  # type: ignore
    correlation, pvalue = pearsonr(golds, probs)

    if return_pvalue:
        return {"pearson_correlation": correlation, "pearson_pvalue": pvalue}

    return {"pearson_correlation": correlation}
