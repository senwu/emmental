import itertools
import logging
import os
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from emmental.meta import Meta
from emmental.task import EmmentalTask
from emmental.utils.utils import construct_identifier, move_to_device, prob_to_pred

logger = logging.getLogger(__name__)


class EmmentalModel(nn.Module):
    """A class to build multi-task model.

    :param name: Name of the model
    :type name: str
    :param tasks: a list of Tasks that trains jointly
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

    def flow(self, X_dict, task_names):
        """Forward based on input and task flow.
        Note: We assume that all shared modules from all tasks are based on the
        same input.

        :param X_dict: The input data
        :type X_dict: dict of tensor
        :param task_names: The task names that needs to forward
        :type task_names: list of str
        :return: The output of all forwarded modules
        :rtype: dict
        """

        X_dict = move_to_device(X_dict, Meta.config["model_config"]["device"])

        output_dict = dict(_input_=X_dict)

        # Call forward for each task
        for task_name in task_names:
            for action in self.task_flows[task_name]:
                if action["name"] not in output_dict:
                    if action["inputs"]:
                        try:
                            input = [
                                output_dict[action_name][output_index]
                                for action_name, output_index in action["inputs"]
                            ]
                        except Exception:
                            raise ValueError(f"Unrecognized action {action}.")
                        output = self.module_pool[action["module"]].forward(*input)
                    else:
                        output = self.module_pool[action["module"]].forward(output_dict)
                    if isinstance(output, tuple):
                        output = list(output)
                    if not isinstance(output, list):
                        output = [output]
                    output_dict[action["name"]] = output

        return output_dict

    def forward(self, uids, X_dict, Y_dict, task_to_label_dict):
        """Calculate the loss, prob for the batch.

        :param uids: The uids of input data
        :type uids: list
        :param X_dict: The input data
        :type X_dict: dict of tensors
        :param Y_dict: The output data
        :type Y_dict: dict of tensors
        :param task_to_label_dict: The task to label mapping
        :type task_to_label_dict: dict
        :return: The (active) uids, loss and prob in the batch of all tasks
        :rtype: dict, dict, dict
        """

        uid_dict = defaultdict(list)
        loss_dict = defaultdict(list)
        prob_dict = defaultdict(list)
        gold_dict = defaultdict(list)

        output_dict = self.flow(X_dict, task_to_label_dict.keys())

        # Calculate loss for each task
        for task_name, label_name in task_to_label_dict.items():
            Y = Y_dict[label_name]

            # Select the active samples
            if len(Y.size()) == 1:
                active = Y.detach() != Meta.config["learner_config"]["ignore_index"]
            else:
                active = torch.any(
                    Y.detach() != Meta.config["learner_config"]["ignore_index"], dim=1
                )

            # Only calculate the loss when active example exists
            if active.any():
                uid_dict[task_name] = [*itertools.compress(uids, active.numpy())]

                loss_dict[task_name] = self.loss_funcs[task_name](
                    output_dict,
                    move_to_device(
                        Y_dict[label_name], Meta.config["model_config"]["device"]
                    ),
                    move_to_device(active, Meta.config["model_config"]["device"]),
                )

                prob_dict[task_name] = (
                    self.output_funcs[task_name](output_dict)[
                        move_to_device(active, Meta.config["model_config"]["device"])
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

                gold_dict[task_name] = Y_dict[label_name][active].cpu().numpy()

        return uid_dict, loss_dict, prob_dict, gold_dict

    @torch.no_grad()
    def predict(self, dataloader, return_preds=False):

        self.eval()

        uid_dict = defaultdict(list)
        gold_dict = defaultdict(list)
        prob_dict = defaultdict(list)
        pred_dict = defaultdict(list)
        loss_dict = defaultdict(float)

        # Collect dataloader information
        task_to_label_dict = dataloader.task_to_label_dict
        uid = dataloader.uid

        for batch_num, (X_bdict, Y_bdict) in enumerate(dataloader):
            uid_bdict, loss_bdict, prob_bdict, gold_bdict = self.forward(
                X_bdict[uid], X_bdict, Y_bdict, task_to_label_dict
            )
            for task_name in uid_bdict.keys():
                uid_dict[task_name].extend(uid_bdict[task_name])
                prob_dict[task_name].extend(prob_bdict[task_name])
                gold_dict[task_name].extend(gold_bdict[task_name])
                loss_dict[task_name] += loss_bdict[task_name].item() * len(
                    uid_bdict[task_name]
                )

        # Calculate average loss
        for task_name in uid_dict.keys():
            loss_dict[task_name] /= len(uid_dict[task_name])

        res = {
            "uids": uid_dict,
            "golds": gold_dict,
            "probs": prob_dict,
            "losses": loss_dict,
        }

        if return_preds:
            for task_name, prob in prob_dict.items():
                pred_dict[task_name] = prob_to_pred(prob)
            res["preds"] = pred_dict

        return res

    @torch.no_grad()
    def score(self, dataloaders, return_average=True):
        """Score the data from dataloader with the model

        :param dataloaders: the dataloader that performs scoring
        :type dataloaders: dataloader
        :param return_average: Whether return average scores
        :type return_average: bool
        """

        self.eval()

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        metric_score_dict = dict()

        if return_average:
            micro_score_dict = defaultdict(list)
            macro_score_dict = defaultdict(list)
            macro_loss_dict = defaultdict(list)

        for dataloader in dataloaders:
            predictions = self.predict(dataloader, return_preds=True)
            for task_name in predictions["golds"].keys():
                metric_score = self.scorers[task_name].score(
                    predictions["golds"][task_name],
                    predictions["probs"][task_name],
                    predictions["preds"][task_name],
                    predictions["uids"][task_name],
                )
                for metric_name, metric_value in metric_score.items():
                    identifier = construct_identifier(
                        task_name, dataloader.data_name, dataloader.split, metric_name
                    )
                    metric_score_dict[identifier] = metric_value

                # Store the loss
                identifier = construct_identifier(
                    task_name, dataloader.data_name, dataloader.split, "loss"
                )
                metric_score_dict[identifier] = predictions["losses"][task_name]

                if return_average:
                    # Collect average score
                    identifier = construct_identifier(
                        task_name, dataloader.data_name, dataloader.split, "average"
                    )
                    metric_score_dict[identifier] = np.mean(list(metric_score.values()))

                    micro_score_dict[dataloader.split].extend(
                        list(metric_score.values())
                    )
                    macro_score_dict[dataloader.split].append(
                        metric_score_dict[identifier]
                    )

                    # Store the loss
                    identifier = construct_identifier(
                        task_name, dataloader.data_name, dataloader.split, "loss"
                    )
                    macro_loss_dict[dataloader.split].append(
                        metric_score_dict[identifier]
                    )

        if return_average:
            # Collect split-wise micro/macro average score
            for split in micro_score_dict.keys():
                identifier = construct_identifier(
                    "model", "all", split, "micro_average"
                )
                metric_score_dict[identifier] = np.mean(micro_score_dict[split])
                identifier = construct_identifier(
                    "model", "all", split, "macro_average"
                )
                metric_score_dict[identifier] = np.mean(macro_score_dict[split])
                identifier = construct_identifier("model", "all", split, "loss")
                metric_score_dict[identifier] = np.mean(macro_loss_dict[split])

            # Collect overall micro/macro average score/loss
            identifier = construct_identifier("model", "all", "all", "micro_average")
            metric_score_dict[identifier] = np.mean(list(micro_score_dict.values()))
            identifier = construct_identifier("model", "all", "all", "macro_average")
            metric_score_dict[identifier] = np.mean(list(macro_score_dict.values()))
            identifier = construct_identifier("model", "all", "all", "loss")
            metric_score_dict[identifier] = np.mean(list(macro_loss_dict.values()))

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
                "module_pool": self.collect_state_dict(),
                # "task_names": self.task_names,
                # "task_flows": self.task_flows,
                # "loss_funcs": self.loss_funcs,
                # "output_funcs": self.output_funcs,
                # "scorers": self.scorers,
            }
        }

        try:
            torch.save(state_dict, model_path)
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model saved in {model_path}")

    def load(self, model_path):
        """Load model state_dict from file and reinitialize the model weights.

        :param model_path: Saved model path.
        :type model_path: str
        """

        if not os.path.exists(model_path):
            logger.error("Loading failed... Model does not exist.")

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        except BaseException:
            logger.error(f"Loading failed... Cannot load model from {model_path}")
            raise

        self.load_state_dict(checkpoint["model"]["module_pool"])

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"[{self.name}] Model loaded from {model_path}")

        # Move model to specified device
        self._move_to_device()

    def collect_state_dict(self):
        state_dict = defaultdict(list)

        for module_name, module in self.module_pool.items():
            if Meta.config["model_config"]["dataparallel"]:
                state_dict[module_name] = module.module.state_dict()
            else:
                state_dict[module_name] = module.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):

        for module_name, module_state_dict in state_dict.items():
            if module_name in self.module_pool:
                if Meta.config["model_config"]["dataparallel"]:
                    self.module_pool[module_name].module.load_state_dict(
                        module_state_dict
                    )
                else:
                    self.module_pool[module_name].load_state_dict(module_state_dict)
            else:
                logger.info(f"Missing {module_name} in module_pool, skip it..")
