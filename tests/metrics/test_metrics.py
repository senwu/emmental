#! /usr/bin/env python

import logging

import numpy as np

from emmental.metrics import METRICS


def test_metrics(caplog):
    """Unit test of meta"""

    caplog.set_level(logging.INFO)

    metrics = METRICS

    golds = np.array([1, 0, 1, 0, 1, 0])
    preds = np.array([1, 1, 1, 1, 1, 0])
    probs = np.array([0.8, 0.6, 0.9, 0.7, 0.7, 0.2])

    res = dict()

    for metric_name, metric in metrics.items():
        res.update(metric(golds, probs, preds))

    assert res == {
        "accuracy": 0.6666666666666666,
        "precision": 0.6,
        "recall": 1.0,
        "f1": 0.7499999999999999,
        "matthews_corrcoef": 0.4472135954999579,
        "mean_squared_error": 0.17166666666666666,
        "pearson_correlation": 0.4472135954999579,
        "pearson_pvalue": 0.37390096630005887,
        "spearman_correlation": 0.447213595499958,
        "spearman_pvalue": 0.3739009663000589,
        "pearson_spearman": 0.447213595499958,
    }
