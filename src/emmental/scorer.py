import logging

from emmental.metrics import METRICS

logger = logging.getLogger(__name__)


class Scorer(object):
    """A class to score tasks

    :param metrics: a list of metric names which provides in emmental (e.g., accuracy)
    :type metrics: list
    :param customize_metric_funcs: a dict of customize metric where key is the metric
    name and value is the metric function which takes gold, preds, probs as input
    :type customize_metric_funcs: dict
    """

    def __init__(self, metrics=[], customize_metric_funcs={}):
        self.metrics = dict()
        for metric in metrics:
            if metric not in METRICS:
                raise ValueError(f"Unrecognized metric: {metric}")
            self.metrics[metric] = METRICS[metric]

        self.metrics.update(customize_metric_funcs)

    def score(self, gold, preds, probs):
        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
            res = metric(gold, preds, probs)

            if isinstance(res, dict):
                metric_dict.update(res)
            else:
                metric_dict[metric_name] = res

        return metric_dict
