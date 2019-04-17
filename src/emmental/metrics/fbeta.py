from emmental.metrics.precision import precision_scorer
from emmental.metrics.recall import recall_scorer


def fbeta_scorer(gold, probs, preds, pos_label=1, beta=1):
    """F-beta score is the weighted harmonic mean of precision and recall.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param pos_label: The positive class label, defaults to 1
    :param pos_label: int, optional
    :param beta: Weight of precision in harmonic mean, defaults to 1
    :param beta: float, optional
    :return: F-beta score.
    :rtype: dict
    """

    precision = precision_scorer(gold, probs, preds, pos_label)["precision"]
    recall = recall_scorer(gold, probs, preds, pos_label)["recall"]

    fbeta = (
        (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        if (beta ** 2 * precision) + recall > 0
        else 0.0
    )

    return {f"f{beta}": fbeta}


def f1_scorer(gold, probs, preds, pos_label=1):
    return {"f1": fbeta_scorer(gold, probs, preds, pos_label, beta=1)["f1"]}
