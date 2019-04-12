import numpy as np
from scipy.stats import spearmanr


def spearman_correlation_scorer(gold, probs, preds):
    correlation, pvalue = spearmanr(gold, preds)
    if np.isnan(correlation):
        correlation = 0.0
    return {"spearman_correlation": correlation, "spearman_pvalue": pvalue}
