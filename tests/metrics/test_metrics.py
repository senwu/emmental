"""Emmental metric unit tests."""
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
from tests.utils import isequal

GOLDS = np.array([0, 1, 0, 1, 0, 1])
PROB_GOLDS = np.array(
    [[0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]]
)
PROBS = np.array(
    [[0.9, 0.1], [0.6, 0.4], [1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.05, 0.95]]
)

UNARY_PROBS = np.array([0.1, 0.4, 0.0, 0.2, 0.4, 0.95])

PREDS = np.array([0, 0, 0, 0, 0, 1])


def test_accuracy(caplog):
    """Unit test of accuracy_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = accuracy_scorer(GOLDS, PROBS, PREDS)

    assert isequal(metric_dict, {"accuracy": 0.6666666666666666})

    metric_dict = accuracy_scorer(GOLDS, None, PREDS)

    assert isequal(metric_dict, {"accuracy": 0.6666666666666666})

    metric_dict = accuracy_scorer(GOLDS, PROBS, None)

    assert isequal(metric_dict, {"accuracy": 0.6666666666666666})

    metric_dict = accuracy_scorer(GOLDS, None, PREDS, normalize=False)

    assert isequal(metric_dict, {"accuracy": 4})

    metric_dict = accuracy_scorer(GOLDS, PROBS, PREDS, topk=2)

    assert isequal(metric_dict, {"accuracy@2": 1.0})

    metric_dict = accuracy_scorer(PROB_GOLDS, None, PREDS)

    assert isequal(metric_dict, {"accuracy": 0.6666666666666666})

    metric_dict = accuracy_scorer(PROB_GOLDS, PROBS, PREDS, topk=2)

    assert isequal(metric_dict, {"accuracy@2": 1.0})

    metric_dict = accuracy_scorer(PROB_GOLDS, PROBS, PREDS, topk=2, normalize=False)

    assert isequal(metric_dict, {"accuracy@2": 6})


def test_precision(caplog):
    """Unit test of precision_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = precision_scorer(GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"precision": 1})

    metric_dict = precision_scorer(GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"precision": 1})

    metric_dict = precision_scorer(GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"precision": 0.6})

    metric_dict = precision_scorer(PROB_GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"precision": 1})

    metric_dict = precision_scorer(PROB_GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"precision": 1})

    metric_dict = precision_scorer(PROB_GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"precision": 0.6})


def test_recall(caplog):
    """Unit test of recall_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = recall_scorer(GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"recall": 0.3333333333333333})

    metric_dict = recall_scorer(GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"recall": 0.3333333333333333})

    metric_dict = recall_scorer(GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"recall": 1})

    metric_dict = recall_scorer(PROB_GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"recall": 0.3333333333333333})

    metric_dict = recall_scorer(PROB_GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"recall": 0.3333333333333333})

    metric_dict = recall_scorer(PROB_GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"recall": 1})


def test_f1(caplog):
    """Unit test of f1_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = f1_scorer(GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"f1": 0.5})

    metric_dict = f1_scorer(GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"f1": 0.5})

    metric_dict = f1_scorer(GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"f1": 0.7499999999999999})

    metric_dict = f1_scorer(PROB_GOLDS, PROBS, PREDS, pos_label=1)
    assert isequal(metric_dict, {"f1": 0.5})

    metric_dict = f1_scorer(PROB_GOLDS, None, PREDS, pos_label=1)
    assert isequal(metric_dict, {"f1": 0.5})

    metric_dict = f1_scorer(PROB_GOLDS, None, PREDS, pos_label=0)
    assert isequal(metric_dict, {"f1": 0.7499999999999999})


def test_fbeta(caplog):
    """Unit test of fbeta_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = fbeta_scorer(GOLDS, PROBS, PREDS, pos_label=1, beta=2)
    assert isequal(metric_dict, {"f2": 0.3846153846153846})

    metric_dict = fbeta_scorer(GOLDS, None, PREDS, pos_label=1, beta=2)
    assert isequal(metric_dict, {"f2": 0.3846153846153846})

    metric_dict = fbeta_scorer(GOLDS, None, PREDS, pos_label=0, beta=2)
    assert isequal(metric_dict, {"f2": 0.8823529411764706})

    metric_dict = fbeta_scorer(PROB_GOLDS, PROBS, PREDS, pos_label=1, beta=2)
    assert isequal(metric_dict, {"f2": 0.3846153846153846})

    metric_dict = fbeta_scorer(PROB_GOLDS, None, PREDS, pos_label=1, beta=2)
    assert isequal(metric_dict, {"f2": 0.3846153846153846})

    metric_dict = fbeta_scorer(PROB_GOLDS, None, PREDS, pos_label=0, beta=2)
    assert isequal(metric_dict, {"f2": 0.8823529411764706})


def test_matthews_corrcoef(caplog):
    """Unit test of matthews_correlation_coefficient_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = matthews_correlation_coefficient_scorer(GOLDS, PROBS, PREDS)
    assert isequal(metric_dict, {"matthews_corrcoef": 0.4472135954999579})

    metric_dict = matthews_correlation_coefficient_scorer(GOLDS, None, PREDS)
    assert isequal(metric_dict, {"matthews_corrcoef": 0.4472135954999579})

    metric_dict = matthews_correlation_coefficient_scorer(PROB_GOLDS, PROBS, PREDS)
    assert isequal(metric_dict, {"matthews_corrcoef": 0.4472135954999579})

    metric_dict = matthews_correlation_coefficient_scorer(PROB_GOLDS, None, PREDS)
    assert isequal(metric_dict, {"matthews_corrcoef": 0.4472135954999579})


def test_mean_squared_error(caplog):
    """Unit test of mean_squared_error_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = mean_squared_error_scorer(GOLDS, UNARY_PROBS, None)
    assert isequal(metric_dict, {"mean_squared_error": 0.1954166666666667})

    metric_dict = mean_squared_error_scorer(PROB_GOLDS, PROBS, None)
    assert isequal(metric_dict, {"mean_squared_error": 0.16708333333333336})


def test_pearson_correlation(caplog):
    """Unit test of pearson_correlation_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = pearson_correlation_scorer(GOLDS, UNARY_PROBS, None)
    assert isequal(metric_dict, {"pearson_correlation": 0.5667402091575048})

    metric_dict = pearson_correlation_scorer(
        GOLDS, UNARY_PROBS, None, return_pvalue=True
    )
    assert isequal(
        metric_dict,
        {
            "pearson_correlation": 0.5667402091575048,
            "pearson_pvalue": 0.24090659530906683,
        },
    )


def test_spearman_correlation(caplog):
    """Unit test of spearman_correlation_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = spearman_correlation_scorer(GOLDS, UNARY_PROBS, None)
    assert isequal(metric_dict, {"spearman_correlation": 0.5940885257860046})

    metric_dict = spearman_correlation_scorer(
        GOLDS, UNARY_PROBS, None, return_pvalue=True
    )
    assert isequal(
        metric_dict,
        {
            "spearman_correlation": 0.5940885257860046,
            "spearman_pvalue": 0.21370636293028789,
        },
    )


def test_pearson_spearman(caplog):
    """Unit test of pearson_spearman_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = pearson_spearman_scorer(GOLDS, UNARY_PROBS, None)
    assert isequal(metric_dict, {"pearson_spearman": 0.5804143674717547})


def test_roc_auc(caplog):
    """Unit test of roc_auc_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = roc_auc_scorer(GOLDS, PROBS, None)

    assert isequal(metric_dict, {"roc_auc": 0.8333333333333333})

    metric_dict = roc_auc_scorer(PROB_GOLDS, PROBS, None)

    assert isequal(metric_dict, {"roc_auc": 0.8333333333333333})

    metric_dict = roc_auc_scorer(GOLDS, UNARY_PROBS, None)

    assert isequal(metric_dict, {"roc_auc": 0.8333333333333334})

    metric_dict = roc_auc_scorer(PROB_GOLDS, UNARY_PROBS, None)

    assert isequal(metric_dict, {"roc_auc": 0.8333333333333334})

    ALL_ONES = np.array([1, 1, 1, 1, 1, 1])

    metric_dict = roc_auc_scorer(ALL_ONES, PROBS, None)
    assert isequal(metric_dict, {"roc_auc": float("nan")})


def test_accuracy_f1(caplog):
    """Unit test of accuracy_f1_scorer."""
    caplog.set_level(logging.INFO)

    metric_dict = accuracy_f1_scorer(GOLDS, None, PREDS)

    assert isequal(metric_dict, {"accuracy_f1": 0.5833333333333333})

    metric_dict = accuracy_f1_scorer(GOLDS, None, PREDS, pos_label=1)

    assert isequal(metric_dict, {"accuracy_f1": 0.5833333333333333})

    metric_dict = accuracy_f1_scorer(GOLDS, None, PREDS, pos_label=0)

    assert isequal(metric_dict, {"accuracy_f1": 0.7083333333333333})

    metric_dict = accuracy_f1_scorer(PROB_GOLDS, None, PREDS)

    assert isequal(metric_dict, {"accuracy_f1": 0.5833333333333333})
