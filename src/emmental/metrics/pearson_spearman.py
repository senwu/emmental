import numpy as np

from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def pearson_spearman_scorer(gold, probs, preds):
    """Average of Pearson correlation coefficient and the p-value and Spearman
    rank-order correlation coefficient.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Pearson correlation coefficient, the p-value and Spearman
    rank-order correlation coefficient and the average.
    :rtype: dict
    """

    metrics = dict()
    metrics.update(pearson_correlation_scorer(gold, probs, preds))
    metrics.update(spearman_correlation_scorer(gold, probs, preds))
    metrics["pearson_spearman"] = np.mean(
        [metrics["pearson_correlation"], metrics["spearman_correlation"]]
    )
    return metrics
