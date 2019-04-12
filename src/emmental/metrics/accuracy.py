import numpy as np


def accuracy_scorer(gold, preds, probs):
    return {"accuracy": np.where(gold == preds)[0].shape[0] / gold.shape[0]}
