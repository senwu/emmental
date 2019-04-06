import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(gold, preds):
    correlation, pvalue = spearmanr(gold, preds)
    if np.isnan(correlation):
        correlation = 0.0
    return {"pearson_correlation": correlation, "pearson_pvalue": pvalue}
