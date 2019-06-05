from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient_scorer(golds, probs, preds, uids=None):
    """Matthews correlation coefficient (MCC).

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :return: Matthews correlation coefficient score.
    :rtype: dict
    """

    return {"matthews_corrcoef": matthews_corrcoef(golds, preds)}
