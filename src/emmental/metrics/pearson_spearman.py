from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def pearson_spearman_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    r"""Average of Pearson correlation coefficient and Spearman rank-order
    correlation coefficient.

    Args:
      golds(ndarray): Ground truth values.
      probs(ndarray): Predicted probabilities.
      preds(ndarray or None): Predicted values.
      uids(list, optional): Unique ids, defaults to None.

    Returns:
      dict: The average of Pearson correlation coefficient and Spearman rank-order
      correlation coefficient.

    """

    metrics = dict()
    pearson_correlation = pearson_correlation_scorer(golds, probs, preds, uids)
    spearman_correlation = spearman_correlation_scorer(golds, probs, preds, uids)
    metrics["pearson_spearman"] = np.mean(
        [
            pearson_correlation["pearson_correlation"],
            spearman_correlation["spearman_correlation"],
        ]
    )
    return metrics
