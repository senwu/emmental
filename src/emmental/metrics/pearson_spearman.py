"""Emmental pearson spearman scorer."""
from statistics import mean
from typing import Dict, List, Optional

from numpy import ndarray

from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def pearson_spearman_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Average of Pearson and Spearman rank-order correlation coefficients.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.

    Returns:
      The average of Pearson correlation coefficient and Spearman rank-order
      correlation coefficient.
    """
    metrics = dict()
    pearson_correlation = pearson_correlation_scorer(golds, probs, preds, uids)
    spearman_correlation = spearman_correlation_scorer(golds, probs, preds, uids)
    metrics["pearson_spearman"] = mean(
        [
            pearson_correlation["pearson_correlation"],
            spearman_correlation["spearman_correlation"],
        ]
    )
    return metrics
