import torch
import torch.nn as nn


class EmmentalModel(nn.Module):
    """A class for multi-task in Emmental MTL model.
    """

    def __init__(self, tasks, name=None):
        super().__init__()
        self.name = name if name is not None else type(self).__name__
        self._build_network(tasks)

    def _build_network(self, tasks):
        self.module_pool = nn.ModuleDict()
        self.task_flows = dict()
        self.loss_funcs = dict()
        self.output_funcs = dict()
        print(type(self.module_pool))
        print(self.module_pool)
        print(self.module_pool.keys())
        for task in tasks:
            # Combine module_pool from all tasks
            for key in task.module_pool.keys():
                if key in self.module_pool.keys():
                    task.module_pool[key] = self.module_pool[key]
                else:
                    self.module_pool[key] = task.module_pool[key]
            # Collect task flows
            self.task_flows[task.name] = task.task_flow
            # Collect loss functions
            self.loss_funcs[task.name] = task.loss_func
            # Collect output functions
            self.output_funcs[task.name] = task.output_func

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

    def forward(self, X, task_name):
        task_flow = self.task_flows[task_name]
        immediate_ouput = [[X]]

        for action in task_flow:
            input = [
                immediate_ouput[action_index][output_index]
                for action_index, output_index in action["inputs"]
            ]
            output = self.module_pool[action["module"]].forward(*input)
            if isinstance(output, tuple):
                output = list(output)
            if not isinstance(output, list):
                output = [output]
            immediate_ouput.append(output)
        return immediate_ouput

    def calc_loss(self, X, task_name):
        pass

    @torch.no_grad()
    def calc_probs(self, X, task_name):
        pass
