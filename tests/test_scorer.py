"""Emmental scorer unit tests."""
import logging

import numpy as np
import pytest

from emmental import Scorer


def test_scorer(caplog):
    """Unit test of scorer."""
    caplog.set_level(logging.INFO)

    golds = np.array([1, 0, 1, 0, 1, 0])
    preds = np.array([1, 1, 1, 1, 1, 0])
    probs = np.array(
        [[0.2, 0.8], [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.3, 0.7], [0.8, 0.2]]
    )

    def sum(gold, probs, preds, uids):
        return np.sum(preds)

    scorer = Scorer(
        metrics=["accuracy", "accuracy@2", "f1"], customize_metric_funcs={"sum": sum}
    )

    assert scorer.score(golds, probs, preds) == {
        "accuracy": 0.6666666666666666,
        "accuracy@2": 1.0,
        "f1": 0.7499999999999999,
        "sum": 5,
    }


def test_scorer_with_unknown_metric(caplog):
    """Unit test of scorer with unknown metric."""
    caplog.set_level(logging.INFO)

    with pytest.raises(ValueError):
        Scorer(metrics=["acc"])


def test_scorer_with_no_gold(caplog):
    """Unit test of scorer with no gold metric."""
    caplog.set_level(logging.INFO)

    preds = np.array([1, 1, 1, 1, 1, 0])

    scorer = Scorer(metrics=["accuracy"])

    score = scorer.score([], None, preds)

    score["accuracy"] == float("nan")


def test_scorer_with_value_error(caplog):
    """Unit test of scorer with no gold metric."""
    caplog.set_level(logging.INFO)

    scorer = Scorer(metrics=["accuracy"])

    with pytest.raises(AttributeError):
        scorer.score("a", [1, 2, 3], [1, 2, 3])

    scorer = Scorer(metrics=["pearson_correlation"])

    with pytest.raises(TypeError):
        scorer.score([1, 2, 3], "a", [1, 2, 3])

    scorer = Scorer(metrics=["matthews_correlation"])

    with pytest.raises(ValueError):
        scorer.score([1, 2, 3], [1, 2, 3], "a")
