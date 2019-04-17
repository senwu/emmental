from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(gold, probs, preds):
    """Mean squared error regression loss.

    :param gold: Ground truth (correct) target values.
    :type gold: 1-d np.array
    :param probs: Predicted target probabilities.
    :type probs: 1-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :return: Mean squared error regression loss.
    :rtype: dict
    """

    return {"mean_squared_error": float(mean_squared_error(gold, probs))}
