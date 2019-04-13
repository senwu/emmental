from abc import ABC

from torch import nn


class Task(ABC):
    """A abstract calss for task in Emmental MTL model.
    """

    def __init__(self, name, module_pool, task_flow, loss_func, output_func, scorer):
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
