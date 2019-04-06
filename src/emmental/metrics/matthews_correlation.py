from sklearn.metrics import matthews_corrcoef


def matthews_correlation_coefficient(gold, preds):
    return {"matthews_corrcoef": matthews_corrcoef(gold, preds)}
