import torch
import torch.nn as nn


class EmmentalModel(nn.Module):
    """A class for multi-task in Emmental MTL model.

    :param tasks: a list of Task that trains jointly
    :type tasks: list of Task project
    :param name: Name of the model
    :type name: str
    """

    def __init__(self, tasks, name=None):
        super().__init__()
        self.name = name if name is not None else type(self).__name__

        # Initiate the model attributes
        self.module_pool = nn.ModuleDict()
        self.task_flows = dict()
        self.loss_funcs = dict()
        self.output_funcs = dict()

        self._build_network(tasks)

    def _build_network(self, tasks):
        """Build the MTL network using all tasks"""
        for task in tasks:
            self._add_task(task)

    def _add_task(self, task):
        """Add a single task into MTL network"""
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

    def _update_task(self, task):
        """Update a existing task in MTL network"""
        # Update module_pool with task
        for key in task.module_pool.keys():
            # Update the model's module with the task's module
            self.module_pool[key] = task.module_pool[key]
        # Update task flows
        self.task_flows[task.name] = task.task_flow
        # Update loss functions
        self.loss_funcs[task.name] = task.loss_func
        # Update output functions
        self.output_funcs[task.name] = task.output_func

    def _remove_task(self, task_name):
        """Remove a existing task from MTL network"""
        if task_name not in self.task_flows:
            print(f"Task ({task_name}) not in the current model, skip...")
            return

        # Remove task by task_name
        print(f"Removing Task ({task_name})..")
        del self.task_flows[task_name]
        del self.loss_funcs[task_name]
        del self.output_funcs[task_name]
        # TODO: remove the modules only associate with that task

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

    def forward(self, X, task_names):
        immediate_ouput_dict = dict()

        # Call forward for each task
        for task_name in task_names:
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
            immediate_ouput_dict[task_name] = immediate_ouput
        return immediate_ouput_dict

    def calculate_loss(self, X, Ys, task_names):
        pass

    @torch.no_grad()
    def calculate_probs(self, X, task_name):
        pass
