from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient_scorer(gold, preds, probs):
    return {"matthews_corrcoef": matthews_corrcoef(gold, preds)}
