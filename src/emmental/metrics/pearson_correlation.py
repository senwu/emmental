import numpy as np
from scipy.stats import pearsonr


def pearson_correlation_scorer(gold, preds, probs):
    correlation, pvalue = pearsonr(gold, preds)
    if np.isnan(correlation):
        correlation = 0.0
    return {"pearson_correlation": correlation, "pearson_pvalue": pvalue}
