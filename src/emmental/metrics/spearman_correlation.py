import numpy as np
from scipy.stats import spearmanr


def spearman_correlation_scorer(golds, probs, preds):
    """Spearman rank-order correlation coefficient and the p-value.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array
    :return: Spearman rank-order correlation coefficient and the p-value
    :rtype: dict
    """

    probs = np.vstack(probs).squeeze()
    correlation, pvalue = spearmanr(golds, probs)
    if np.isnan(correlation):
        correlation = 0.0
    return {"spearman_correlation": correlation, "spearman_pvalue": pvalue}
