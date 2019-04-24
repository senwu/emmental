import logging
import os
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from emmental.meta import Meta
from emmental.task import EmmentalTask
from emmental.utils.utils import move_to_device, prob_to_pred

logger = logging.getLogger(__name__)


class EmmentalModel(nn.Module):
    """A class to build multi-task model.

    :param name: Name of the model
    :type name: str
    :param tasks: a list of Task that trains jointly
    :type tasks: list of Task project
    """

    def __init__(self, name, tasks=None):
        super().__init__()
        self.name = name if name is not None else type(self).__name__

        # Initiate the model attributes
        self.module_pool = nn.ModuleDict()
        self.task_names = set()
        self.task_flows = dict()
        self.loss_funcs = dict()
        self.output_funcs = dict()
        self.scorers = dict()

        # Build network with given tasks
        if tasks is not None:
            self._build_network(tasks)

        if Meta.config["meta_config"]["verbose"]:
            logger.info(
                f"Created emmental model {self.name} that contains "
                f"task {self.task_names}."
            )

        # Move model to specified device
        self._move_to_device()

    def _move_to_device(self):
        """Move model to specified device."""

        if Meta.config["meta_config"]["device"] >= 0:
            if torch.cuda.is_available():
                if Meta.config["meta_config"]["verbose"]:
                    logger.info("Moving model to GPU.")
                self.to(torch.device(f"cuda:{Meta.config['meta_config']['device']}"))
            else:
                if Meta.config["meta_config"]["verbose"]:
                    logger.info("No cuda device available. Switch to cpu instead.")

    def _build_network(self, tasks):
        """Build the MTL network using all tasks"""
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        for task in tasks:
            if task.name in self.task_names:
                raise ValueError(
                    f"Found duplicate task {task.name}, different task should use "
                    f"different task name."
                )
            if not isinstance(task, EmmentalTask):
                raise ValueError(f"Unrecognized task type {task}.")
            self._add_task(task)

    def _add_task(self, task):
        """Add a single task into MTL network"""
        # Combine module_pool from all tasks
        for key in task.module_pool.keys():
            if key in self.module_pool.keys():
                task.module_pool[key] = self.module_pool[key]
            else:
                self.module_pool[key] = task.module_pool[key]
        # Collect task names
        self.task_names.add(task.name)
        # Collect task flows
        self.task_flows[task.name] = task.task_flow
        # Collect loss functions
        self.loss_funcs[task.name] = task.loss_func
        # Collect output functions
        self.output_funcs[task.name] = task.output_func
        # Collect scorers
        self.scorers[task.name] = task.scorer

        # Move model to specified device
        self._move_to_device()

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
        # Collect scorers
        self.scorers[task.name] = task.scorer

        # Move model to specified device
        self._move_to_device()

    def _remove_task(self, task_name):
        """Remove a existing task from MTL network"""
        if task_name not in self.task_flows:
            if Meta.config["meta_config"]["verbose"]:
                logger.info(f"Task ({task_name}) not in the current model, skip...")
            return

        # Remove task by task_name
        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"Removing Task {task_name}.")
        self.task_names.remove(task_name)
        del self.task_flows[task_name]
        del self.loss_funcs[task_name]
        del self.output_funcs[task_name]
        del self.scorers[task_name]
        # TODO: remove the modules only associate with that task

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

    def forward(self, X_dict, task_names):
        """Calculate the loss given the features and labels

        :param X_dict:
        :type X_dict:
        :param task_names:
        :type task_names:
        """
        immediate_ouput_dict = dict()

        X_dict = move_to_device(X_dict, Meta.config["meta_config"]["device"])

        # Call forward for each task
        for task_name in task_names:
            task_flow = self.task_flows[task_name]
            immediate_ouput = [X_dict]

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

    def calculate_losses(
        self, X_dict, Y_dict, task_names, data_names, label_names, split
    ):
        """Calculate the loss given the features and labels

        :param X_dict:
        :type X_dict:
        :param Y_dict:
        :type Y_dict:
        :param task_names:
        :type task_names:
        """

        loss_dict = dict()
        count_dict = dict()

        immediate_ouput_dict = self.forward(X_dict, task_names)

        # Calculate loss for each task
        for task_name, data_name, label_name in zip(
            task_names, data_names, label_names
        ):
            identifier = "/".join([task_name, data_name, split, "loss"])
            loss_dict[identifier] = self.loss_funcs[task_name](
                immediate_ouput_dict[task_name],
                move_to_device(
                    Y_dict[label_name], Meta.config["meta_config"]["device"]
                ),
            )

            count_dict[identifier] = Y_dict[label_name].size(0)

        return loss_dict, count_dict

    @torch.no_grad()
    def _calculate_probs(self, X_dict, task_names):
        """Calculate the probs given the features

        :param X_dict:
        :type X_dict:
        :param task_names:
        :type task_names:
        """
        prob_dict = dict()

        immediate_ouput_dict = self.forward(X_dict, task_names)

        # Calculate prediction for each task
        for task_name in task_names:
            prob_dict[task_name] = (
                self.output_funcs[task_name](immediate_ouput_dict[task_name])
                .cpu()
                .numpy()
            )

        return prob_dict

    @torch.no_grad()
    def predict(self, dataloader, return_preds=False):

        gold_dict = defaultdict(list)
        prob_dict = defaultdict(list)

        for batch_num, (X_batch_dict, Y_batch_dict) in enumerate(dataloader):
            prob_batch_dict = self._calculate_probs(
                X_batch_dict, [dataloader.task_name]
            )
            for task_name, prob_batch in prob_batch_dict.items():
                prob_dict[task_name].extend(prob_batch)
            gold_dict[task_name].extend(
                Y_batch_dict[dataloader.label_name].cpu().numpy()
            )
        for task_name in gold_dict:
            gold_dict[task_name] = np.array(gold_dict[task_name]).reshape(-1)
            prob_dict[task_name] = np.array(prob_dict[task_name])

        if return_preds:
            pred_dict = defaultdict(list)
            for task_name, prob in prob_dict.items():
                pred_dict[task_name] = prob_to_pred(prob)

        if return_preds:
            return gold_dict, prob_dict, pred_dict
        else:
            return gold_dict, prob_dict

    @torch.no_grad()
    def score(self, dataloaders):
        """Score the data from dataloader with the model

        :param dataloaders: the dataloader that performs scoring
        :type dataloaders: dataloader
        """

        self.eval()

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        metric_score_dict = dict()

        for dataloader in dataloaders:
            gold_dict, prob_dict, pred_dict = self.predict(
                dataloader, return_preds=True
            )
            for task_name in gold_dict.keys():
                metric_score = self.scorers[task_name].score(
                    gold_dict[task_name], prob_dict[task_name], pred_dict[task_name]
                )
                for metric_name, metric_value in metric_score.items():
                    identifier = "/".join(
                        [task_name, dataloader.data_name, dataloader.split, metric_name]
                    )
                    metric_score_dict[identifier] = metric_value

        return metric_score_dict

    def save(self, model_file, save_dir):
        """Save the current model
        :param model_file: Saved model file name.
        :type model_file: str
        :param save_dir: Saved model directory.
        :type save_dir: str
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
            logger.warning("Saving failed... continuing anyway.")

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model saved as {model_file} in {save_dir}")

    def load(self, model_file, save_dir):
        """Load model from file and rebuild the model.
        :param model_file: Saved model file name.
        :type model_file: str
        :param save_dir: Saved model directory.
        :type save_dir: str
        """

        if not os.path.exists(save_dir):
            logger.error("Loading failed... Directory does not exist.")

        try:
            checkpoint = torch.load(f"{save_dir}/{model_file}")
        except BaseException:
            logger.error(
                f"Loading failed... Cannot load model from {save_dir}/{model_file}"
            )

        self.name = checkpoint["name"]
        self.module_pool = checkpoint["module_pool"]
        self.task_flows = checkpoint["task_flows"]
        self.loss_funcs = checkpoint["loss_funcs"]
        self.output_funcs = checkpoint["output_funcs"]

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model loaded as {model_file} in {save_dir}")
