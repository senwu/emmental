from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from emmental.metrics.accuracy import accuracy_scorer
from emmental.metrics.fbeta import f1_scorer


def accuracy_f1_scorer(
    golds: ndarray,
    probs: Optional[ndarray],
    preds: ndarray,
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    r"""Average of accuracy and f1 score.

    Args:
      golds(np.array): Ground truth values.
      probs(np.array or None): Predicted probabilities.
      preds(np.array): Predicted values.
      uids(list, optional): Unique ids, defaults to None.

    Returns:
      dict: Average of accuracy and f1.

    """

    metrics = dict()
    accuracy = accuracy_scorer(golds, probs, preds, uids)
    f1 = f1_scorer(golds, probs, preds, uids)
    metrics["accuracy_f1"] = np.mean([accuracy["accuracy"], f1["f1"]])

    return metrics
