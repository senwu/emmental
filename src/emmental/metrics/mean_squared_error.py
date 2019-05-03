from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(golds, probs, preds):
    """Mean squared error regression loss.

    :param golds: Ground truth (correct) target values.
    :type golds: k-d np.array
    :param probs: Predicted target probabilities.
    :type probs: k-d np.array
    :param preds: Predicted target values. (Not used!)
    :type preds: any
    :return: Mean squared error regression loss.
    :rtype: dict
    """

    return {"mean_squared_error": float(mean_squared_error(golds, probs))}
