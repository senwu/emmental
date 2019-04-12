from emmental.metrics.precision import precision_scorer
from emmental.metrics.recall import recall_scorer


def fbeta_scorer(gold, preds, probs, pos_label=1, beta=1):
    precision = precision_scorer(gold, preds, probs, pos_label)["precision"]
    recall = recall_scorer(gold, preds, probs, pos_label)["recall"]

    fbeta = (
        (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        if (beta ** 2 * precision) + recall > 0
        else 0.0
    )

    return {f"f{beta}": fbeta}


def f1_scorer(gold, preds, probs, pos_label=1):
    return {"f1": fbeta_scorer(gold, preds, probs, pos_label, beta=1)["f1"]}
