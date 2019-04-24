import numpy as np
from scipy.stats import pearsonr


def pearson_correlation_scorer(gold, probs, preds):
    """Pearson correlation coefficient and the p-value.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Pearson correlation coefficient and the p-value.
    :rtype: dict
    """

    probs = np.vstack(probs).squeeze()
    correlation, pvalue = pearsonr(gold, probs)
    if np.isnan(correlation):
        correlation = 0.0
    return {"pearson_correlation": correlation, "pearson_pvalue": pvalue}
