from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient_scorer(gold, probs, preds):
    """Matthews correlation coefficient (MCC).

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Matthews correlation coefficient score.
    :rtype: dict
    """

    return {"matthews_corrcoef": matthews_corrcoef(gold, preds)}
