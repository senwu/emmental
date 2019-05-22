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

    def __init__(self, name=None, tasks=None):
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

        if Meta.config["model_config"]["device"] >= 0:
            if torch.cuda.is_available():
                if Meta.config["meta_config"]["verbose"]:
                    logger.info(
                        f"Moving model to GPU "
                        f"(cuda:{Meta.config['model_config']['device']})."
                    )
                self.to(torch.device(f"cuda:{Meta.config['model_config']['device']}"))
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
            self.add_task(task)

    def add_task(self, task):
        """Add a single task into MTL network"""

        # Combine module_pool from all tasks
        for key in task.module_pool.keys():
            if key in self.module_pool.keys():
                if Meta.config["model_config"]["dataparallel"]:
                    task.module_pool[key] = nn.DataParallel(self.module_pool[key])
                else:
                    task.module_pool[key] = self.module_pool[key]
            else:
                if Meta.config["model_config"]["dataparallel"]:
                    self.module_pool[key] = nn.DataParallel(task.module_pool[key])
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

    def update_task(self, task):
        """Update a existing task in MTL network"""

        # Update module_pool with task
        for key in task.module_pool.keys():
            # Update the model's module with the task's module
            if Meta.config["model_config"]["dataparallel"]:
                self.module_pool[key] = nn.DataParallel(task.module_pool[key])
            else:
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

    def remove_task(self, task_name):
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
        """Forward based on input and task
        Note: We assume that all shared the modules from all tasks are based on the
        the same input.

        :param X_dict: The input data
        :type X_dict: dict of tensor
        :param task_names: The task names that needs to forward
        :type task_names: list of str
        :return: The output of all forwarded modules
        :rtype: dict
        """

        X_dict = move_to_device(X_dict, Meta.config["model_config"]["device"])

        immediate_ouput_dict = dict()
        immediate_ouput_dict["_input_"] = X_dict

        # Call forward for each task
        for task_name in task_names:
            task_flow = self.task_flows[task_name]

            for action in task_flow:
                if action["name"] not in immediate_ouput_dict:
                    try:
                        input = [
                            immediate_ouput_dict[action_name][output_index]
                            for action_name, output_index in action["inputs"]
                        ]
                    except Exception:
                        raise ValueError(f"Unrecognized action {action}.")
                    output = self.module_pool[action["module"]].forward(*input)
                    if isinstance(output, tuple):
                        output = list(output)
                    if not isinstance(output, list):
                        output = [output]
                    immediate_ouput_dict[action["name"]] = output

        return immediate_ouput_dict

    def calculate_loss(self, X_dict, Y_dict, task_to_label_dict, data_name, split):
        """Calculate the loss

        :param X_dict: The input data
        :type X_dict: dict of tensors
        :param Y_dict: The output data
        :type Y_dict: dict of tensors
        :param task_to_label_dict: The task to label mapping
        :type task_to_label_dict: dict
        :param data_name: The dataset name
        :type data_name: str
        :param split: The data split
        :type split: str
        :return: The loss and the number of samples in the batch of all tasks
        :rtype: dict, dict
        """

        loss_dict = dict()
        count_dict = dict()

        immediate_ouput_dict = self.forward(X_dict, task_to_label_dict.keys())

        # Calculate loss for each task
        for task_name, label_name in task_to_label_dict.items():
            identifier = "/".join([task_name, data_name, split, "loss"])

            Y = Y_dict[label_name]

            # Select the active samples
            if len(Y.size()) == 1:
                active = Y.detach() != Meta.config["learner_config"]["ignore_index"]
            else:
                active = torch.any(
                    Y.detach() != Meta.config["learner_config"]["ignore_index"], dim=1
                )

            # Only calculate the loss when active example exists
            if 1 in active:
                count_dict[identifier] = active.sum().item()

                loss_dict[identifier] = self.loss_funcs[task_name](
                    immediate_ouput_dict,
                    move_to_device(
                        Y_dict[label_name], Meta.config["model_config"]["device"]
                    ),
                    move_to_device(active, Meta.config["model_config"]["device"]),
                )

        return loss_dict, count_dict

    @torch.no_grad()
    def _calculate_probs(self, X_dict, task_names):
        """Calculate the probs given the features

        :param X_dict: The input data
        :type X_dict: dict of tensor
        :param task_names: The task names that needs to forward
        :type task_names: list of str
        """

        prob_dict = dict()

        immediate_ouput_dict = self.forward(X_dict, task_names)

        # Calculate prediction for each task
        for task_name in task_names:
            prob_dict[task_name] = (
                self.output_funcs[task_name](immediate_ouput_dict).cpu().numpy()
            )

        return prob_dict

    @torch.no_grad()
    def predict(self, dataloader, return_preds=False):

        gold_dict = defaultdict(list)
        prob_dict = defaultdict(list)

        for batch_num, (X_batch_dict, Y_batch_dict) in enumerate(dataloader):
            prob_batch_dict = self._calculate_probs(
                X_batch_dict, dataloader.task_to_label_dict.keys()
            )
            for task_name in dataloader.task_to_label_dict.keys():
                prob_dict[task_name].extend(prob_batch_dict[task_name])
                gold_dict[task_name].extend(
                    Y_batch_dict[dataloader.task_to_label_dict[task_name]].cpu().numpy()
                )
        for task_name in gold_dict:
            gold_dict[task_name] = np.array(gold_dict[task_name]).reshape(-1)
            prob_dict[task_name] = np.array(prob_dict[task_name])
            active = (
                gold_dict[task_name] != Meta.config["learner_config"]["ignore_index"]
            ).reshape(-1)
            if 0 in active:
                gold_dict[task_name] = gold_dict[task_name][active]
                prob_dict[task_name] = prob_dict[task_name][active]

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
                # import pdb; pdb.set_trace()
                metric_score = self.scorers[task_name].score(
                    gold_dict[task_name], prob_dict[task_name], pred_dict[task_name]
                )
                for metric_name, metric_value in metric_score.items():
                    identifier = "/".join(
                        [task_name, dataloader.data_name, dataloader.split, metric_name]
                    )
                    metric_score_dict[identifier] = metric_value

        # TODO: have a better to handle global evaluation metric
        if Meta.config["learner_config"]["global_evaluation_metric_dict"]:
            global_evaluation_metric_dict = Meta.config["learner_config"][
                "global_evaluation_metric_dict"
            ]
            for metric_name, metric in global_evaluation_metric_dict.items():
                metric_score_dict[metric_name] = metric(metric_score_dict)

        return metric_score_dict

    def save(self, model_path):
        """Save the current model
        :param model_path: Saved model path.
        :type model_path: str
        """

        # Check existence of model saving directory and create if does not exist.
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        state_dict = {
            "model": {
                "name": self.name,
                "module_pool": self.module_pool,
                "task_names": self.task_names,
                "task_flows": self.task_flows,
                "loss_funcs": self.loss_funcs,
                "output_funcs": self.output_funcs,
                "scorers": self.scorers,
            }
        }

        try:
            torch.save(state_dict, model_path)
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model saved in {model_path}")

    def load(self, model_path):
        """Load model from file and rebuild the model.
        :param model_path: Saved model path.
        :type model_path: str
        """

        if not os.path.exists(model_path):
            logger.error("Loading failed... Model does not exist.")

        try:
            checkpoint = torch.load(model_path)
        except BaseException:
            logger.error(f"Loading failed... Cannot load model from {model_path}")

        self.name = checkpoint["model"]["name"]
        self.module_pool = checkpoint["model"]["module_pool"]
        self.task_names = checkpoint["model"]["task_names"]
        self.task_flows = checkpoint["model"]["task_flows"]
        self.loss_funcs = checkpoint["model"]["loss_funcs"]
        self.output_funcs = checkpoint["model"]["output_funcs"]
        self.scorers = checkpoint["model"]["scorers"]

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model loaded from {model_path}")

        # Move model to specified device
        self._move_to_device()
