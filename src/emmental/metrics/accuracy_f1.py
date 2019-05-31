import numpy as np

from emmental.metrics.accuracy import accuracy_scorer
from emmental.metrics.fbeta import f1_scorer


def accuracy_f1_scorer(golds, probs, preds, uids=None):
    """Average of accuracy and f1 score.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :return: Average of accuracy and f1.
    :rtype: dict
    """

    metrics = dict()
    metrics.update(accuracy_scorer(golds, probs, preds, uids))
    metrics.update(f1_scorer(golds, probs, preds, uids))
    metrics["accuracy_f1"] = np.mean([metrics["accuracy"], metrics["f1"]])

    return metrics
