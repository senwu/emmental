import logging
from functools import partial
from typing import Callable, Dict, List

from numpy import ndarray

from emmental.metrics import METRICS
from emmental.utils.utils import array_to_numpy

logger = logging.getLogger(__name__)


class Scorer(object):
    r"""A class to score tasks.

    Args:
      metrics(list): a list of metric names which provides
        in emmental (e.g., accuracy), defaults to [].
      customize_metric_funcs(dict): a dict of customize metric where key is the metric
        name and value is the metric function which takes gold, preds, probs, uids as
        input, defaults to {}.

    """

    def __init__(
        self, metrics: List[str] = [], customize_metric_funcs: Dict[str, Callable] = {}
    ) -> None:
        self.metrics: Dict[str, Callable] = dict()
        for metric in metrics:
            if metric in METRICS:
                self.metrics[metric] = METRICS[metric]  # type: ignore
            elif metric.startswith("accuracy@"):
                self.metrics[metric] = partial(
                    METRICS["accuracy"], topk=int(metric.split("@")[1])  # type: ignore
                )
            else:
                raise ValueError(f"Unrecognized metric: {metric}")

        self.metrics.update(customize_metric_funcs)

    def score(
        self, golds: ndarray, preds: ndarray, probs: ndarray, uids: List[str] = None
    ) -> Dict[str, float]:
        """Calculate the score.

        Args:
          golds(ndarray): Ground truth values.
          probs(ndarray): Predicted probabilities.
          preds(ndarray): Predicted values.
          uids(list, optional): Unique ids, defaults to None.

        Returns:
          dict: score dict.

        """

        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
            # handle no examples
            if len(golds) == 0:
                metric_dict[metric_name] = float("nan")
                continue

            golds = array_to_numpy(golds)
            preds = array_to_numpy(preds)
            probs = array_to_numpy(probs)

            res = metric(golds, preds, probs, uids)

            if isinstance(res, dict):
                metric_dict.update(res)
            else:
                metric_dict[metric_name] = res

        return metric_dict
