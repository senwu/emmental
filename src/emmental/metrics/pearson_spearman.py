import numpy as np

from emmental.metrics.pearson_correlation import pearson_correlation
from emmental.metrics.spearman_correlation import spearman_correlation


def pearson_spearman(gold, preds):
    metrics = dict()
    metrics.update(pearson_correlation(gold, preds))
    metrics.update(spearman_correlation(gold, preds))
    metrics["pearson_spearman"] = np.mean(
        [metrics["pearson_correlation"], metrics["spearman_correlation"]]
    )
    return metrics
