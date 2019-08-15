import numpy as np

from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def pearson_spearman_scorer(golds, probs, preds, uids=None):
    """Average of Pearson correlation coefficient and Spearman rank-order
    correlation coefficient.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :return: Pearson correlation coefficient, the p-value and Spearman
        rank-order correlation coefficient and the average.
    :rtype: dict
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
