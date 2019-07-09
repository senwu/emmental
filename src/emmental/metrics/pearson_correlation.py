import numpy as np
from scipy.stats import pearsonr


def pearson_correlation_scorer(golds, probs, preds, uids=None, return_pvalue=False):
    """Pearson correlation coefficient and the p-value.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :para return_pvalue: Whether return pvalue.
    :type return_pvalue: bool
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
