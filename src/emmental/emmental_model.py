import logging
import os

import torch
import torch.nn as nn


class EmmentalModel(nn.Module):
    """A class for multi-task in Emmental MTL model.

    :param name: Name of the model
    :type name: str
    :param tasks: a list of Task that trains jointly
    :type tasks: list of Task project
    """

    def __init__(self, name=None, tasks=None):
        self.logger = logging.getLogger(__name__)

        super().__init__()
        self.name = name if name is not None else type(self).__name__

        # Initiate the model attributes
        self.module_pool = nn.ModuleDict()
        self.task_flows = dict()
        self.loss_funcs = dict()
        self.output_funcs = dict()

        # Build network with given tasks
        if tasks is not None:
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
        """Calculate the loss given the features and labels

        :param X:
        :type X:
        :param Y_dict:
        :type Y_dict:
        :param task_names:
        :type task_names:
        """
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

    def calculate_losses(self, X, Y_dict, task_names):
        """Calculate the loss given the features and labels

        :param X:
        :type X:
        :param Y_dict:
        :type Y_dict:
        :param task_names:
        :type task_names:
        """
        loss_dict = dict()

        immediate_ouput_dict = self.forward(X, task_names)

        # Calculate loss for each task
        for task_name in task_names:
            loss_dict[task_name] = self.loss_funcs[task_name](
                immediate_ouput_dict[task_name], Y_dict[task_name]
            )

        return loss_dict

    @torch.no_grad()
    def calculate_preds(self, X, task_names):
        """Calculate the loss given the features and labels

        :param X:
        :type X:
        :param Y_dict:
        :type Y_dict:
        :param task_names:
        :type task_names:
        """
        pred_dict = dict()

        immediate_ouput_dict = self.forward(X, task_names)

        # Calculate prediction for each task
        for task_name in task_names:
            pred_dict[task_name] = self.output_funcs[task_name](
                immediate_ouput_dict[task_name]
            )

        return pred_dict

    def save(self, model_file, save_dir, verbose=True):
        """Save the current model
        :param model_file: Saved model file name.
        :type model_file: str
        :param save_dir: Saved model directory.
        :type save_dir: str
        :param verbose: Print log or not
        :type verbose: bool
        """

        # Check existence of model saving directory and create if does not exist.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        params = {
            "name": self.name,
            "module_pool": self.module_pool,
            "task_flows": self.task_flows,
            "loss_funcs": self.loss_funcs,
            "output_funcs": self.output_funcs,
        }

        try:
            torch.save(params, f"{save_dir}/{model_file}")
        except BaseException:
            self.logger.warning("Saving failed... continuing anyway.")

        if verbose:
            self.logger.info(f"[{self.name}] Model saved as {model_file} in {save_dir}")

    def load(self, model_file, save_dir, verbose=True):
        """Load model from file and rebuild the model.
        :param model_file: Saved model file name.
        :type model_file: str
        :param save_dir: Saved model directory.
        :type save_dir: str
        :param verbose: Print log or not
        :type verbose: bool
        """

        if not os.path.exists(save_dir):
            self.logger.error("Loading failed... Directory does not exist.")

        try:
            checkpoint = torch.load(f"{save_dir}/{model_file}")
        except BaseException:
            self.logger.error(
                f"Loading failed... Cannot load model from {save_dir}/{model_file}"
            )

        self.name = checkpoint["name"]
        self.module_pool = checkpoint["module_pool"]
        self.task_flows = checkpoint["task_flows"]
        self.loss_funcs = checkpoint["loss_funcs"]
        self.output_funcs = checkpoint["output_funcs"]

        if verbose:
            self.logger.info(
                f"[{self.name}] Model loaded as {model_file} in {save_dir}"
            )
