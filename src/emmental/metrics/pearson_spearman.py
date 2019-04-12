import numpy as np

from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def pearson_spearman_scorer(gold, preds, probs):
    metrics = dict()
    metrics.update(pearson_correlation_scorer(gold, preds, probs))
    metrics.update(spearman_correlation_scorer(gold, preds, probs))
    print(metrics)
    metrics["pearson_spearman"] = np.mean(
        [metrics["pearson_correlation"], metrics["spearman_correlation"]]
    )
    return metrics
