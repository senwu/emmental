from sklearn.metrics import mean_squared_error


def mean_squared_error_scorer(gold, probs, preds):
    return {"mean_squared_error": float(mean_squared_error(gold, probs))}
