from abc import ABC

from torch import nn


class Task(ABC):
    """A abstract calss for task in Emmental MTL model.
    """

    def __init__(self, name, modules, task_flow, loss_func, output_func):
        self.name = name
        assert isinstance(modules, nn.ModuleDict) is True
        self.modules = modules
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"


class MultiTask(ABC):
    """A abstract calss for multi-task in Emmental MTL model.
    """

    def __init__(self, tasks, name=None):
        self.name = name if name is not None else type(self).__name__
        self.modules = nn.ModuleDict()
        self.task_flows = dict()
        self.loss_funcs = dict()
        self.output_funcs = dict()

        for task in tasks:
            # Combine modules from all tasks
            for key in task.modules.keys():
                if key in self.modules:
                    task.modules[key] = self.modules[key]
                else:
                    self.modules[key] = task.modules[key]
            # Collect task flows
            self.task_flows[task.name] = task.task_flow
            # Collect loss functions
            self.loss_funcs[task.name] = task.loss_func
            # Collect output functions
            self.output_funcs[task.name] = task.output_func

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
