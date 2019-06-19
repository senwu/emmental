import logging

import numpy as np

from emmental.metrics.accuracy import accuracy_scorer
from emmental.metrics.accuracy_f1 import accuracy_f1_scorer
from emmental.metrics.fbeta import f1_scorer, fbeta_scorer
from emmental.metrics.matthews_correlation import (
    matthews_correlation_coefficient_scorer,
)
from emmental.metrics.mean_squared_error import mean_squared_error_scorer
from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.pearson_spearman import pearson_spearman_scorer
from emmental.metrics.precision import precision_scorer
from emmental.metrics.recall import recall_scorer
from emmental.metrics.roc_auc import roc_auc_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer


def isequal(dict_a, dict_b, precision=1e-10):
    for key in dict_a:
        if key not in dict_b or abs(dict_a[key] - dict_b[key]) > precision:
            return False

    for key in dict_b:
        if key not in dict_a or abs(dict_a[key] - dict_b[key]) > precision:
            return False

    return True


def test_accuracy(caplog):
    """Unit test of accuracy_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = accuracy_scorer(golds, None, preds)

    assert isequal(metric_dict, {"accuracy": 0.6666666666666666})


def test_precision(caplog):
    """Unit test of precision_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = precision_scorer(golds, None, preds, pos_label=1)
    assert isequal(metric_dict, {"precision": 1})

    metric_dict = precision_scorer(golds, None, preds, pos_label=2)
    assert isequal(metric_dict, {"precision": 0.6})


def test_recall(caplog):
    """Unit test of recall_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = recall_scorer(golds, None, preds, pos_label=1)
    assert isequal(metric_dict, {"recall": 0.3333333333333333})

    metric_dict = recall_scorer(golds, None, preds, pos_label=2)
    assert isequal(metric_dict, {"recall": 1})


def test_f1(caplog):
    """Unit test of f1_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = f1_scorer(golds, None, preds, pos_label=1)
    assert isequal(metric_dict, {"f1": 0.5})

    metric_dict = f1_scorer(golds, None, preds, pos_label=2)
    assert isequal(metric_dict, {"f1": 0.7499999999999999})


def test_fbeta(caplog):
    """Unit test of fbeta_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = fbeta_scorer(golds, None, preds, pos_label=1, beta=2)
    assert isequal(metric_dict, {"f2": 0.3846153846153846})

    metric_dict = fbeta_scorer(golds, None, preds, pos_label=2, beta=2)
    assert isequal(metric_dict, {"f2": 0.8823529411764706})


def test_matthews_corrcoef(caplog):
    """Unit test of matthews_correlation_coefficient_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = matthews_correlation_coefficient_scorer(golds, None, preds)
    assert isequal(metric_dict, {"matthews_corrcoef": 0.4472135954999579})


def test_mean_squared_error(caplog):
    """Unit test of mean_squared_error_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([3, -0.5, 2, 7])
    probs = np.array([2.5, 0.0, 2, 8])

    metric_dict = mean_squared_error_scorer(golds, probs, None)
    assert isequal(metric_dict, {"mean_squared_error": 0.375})

    golds = np.array([[0.5, 1], [-1, 1], [7, -6]])
    probs = np.array([[0, 2], [-1, 2], [8, -5]])

    metric_dict = mean_squared_error_scorer(golds, probs, None)
    assert isequal(metric_dict, {"mean_squared_error": 0.7083333333333334})


def test_pearson_correlation(caplog):
    """Unit test of pearson_correlation_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([1, 0, 1, 0, 1, 0])
    probs = np.array([0.8, 0.6, 0.9, 0.7, 0.7, 0.2])

    metric_dict = pearson_correlation_scorer(golds, probs, None)
    assert isequal(
        metric_dict,
        {
            "pearson_correlation": 0.6764814252025461,
            "pearson_pvalue": 0.14006598491201774,
        },
    )


def test_spearman_correlation(caplog):
    """Unit test of spearman_correlation_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([1, 0, 1, 0, 1, 0])
    probs = np.array([0.8, 0.6, 0.9, 0.7, 0.7, 0.2])

    metric_dict = spearman_correlation_scorer(golds, probs, None)
    assert isequal(
        metric_dict,
        {
            "spearman_correlation": 0.7921180343813395,
            "spearman_pvalue": 0.06033056705743058,
        },
    )


def test_pearson_spearman(caplog):
    """Unit test of pearson_spearman_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([1, 0, 1, 0, 1, 0])
    probs = np.array([0.8, 0.6, 0.9, 0.7, 0.7, 0.2])

    metric_dict = pearson_spearman_scorer(golds, probs, None)

    assert isequal(
        metric_dict,
        {
            "pearson_correlation": 0.6764814252025461,
            "pearson_pvalue": 0.14006598491201774,
            "spearman_correlation": 0.7921180343813395,
            "spearman_pvalue": 0.06033056705743058,
            "pearson_spearman": 0.7342997297919428,
        },
    )


def test_roc_auc(caplog):
    """Unit test of roc_auc_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    probs = np.array(
        [[0.2, 0.8], [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.3, 0.7], [0.8, 0.2]]
    )

    metric_dict = roc_auc_scorer(golds, probs, None)

    assert isequal(metric_dict, {"roc_auc": 0.9444444444444444})


def test_accuracy_f1(caplog):
    """Unit test of accuracy_f1_scorer"""

    caplog.set_level(logging.INFO)

    golds = np.array([2, 1, 2, 1, 2, 1])
    preds = np.array([2, 2, 2, 2, 2, 1])

    metric_dict = accuracy_f1_scorer(golds, None, preds)

    assert isequal(
        metric_dict,
        {"accuracy": 0.6666666666666666, "f1": 0.5, "accuracy_f1": 0.5833333333333333},
    )
