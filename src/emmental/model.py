"""Emmental model."""
import importlib
import itertools
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor, nn
from torch.nn import ModuleDict

from emmental.data import EmmentalDataLoader
from emmental.meta import Meta
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from emmental.utils.utils import (
    array_to_numpy,
    construct_identifier,
    move_to_device,
    prob_to_pred,
)

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmmentalModel(nn.Module):
    """A class to build multi-task model.

    Args:
      name: Name of the model, defaults to None.
      tasks: A task or a list of tasks.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tasks: Optional[Union[EmmentalTask, List[EmmentalTask]]] = None,
    ) -> None:
        """Initialize EmmentalModel."""
        super().__init__()
        self.name = name if name is not None else type(self).__name__

        # Initiate the model attributes
        self.module_pool: ModuleDict = ModuleDict()
        self.task_names: Set[str] = set()
        self.task_flows: Dict[str, Any] = dict()  # TODO: make it concrete
        self.loss_funcs: Dict[str, Callable] = dict()
        self.output_funcs: Dict[str, Callable] = dict()
        self.scorers: Dict[str, Scorer] = dict()
        self.action_outputs: Dict[
            str, Optional[List[Union[Tuple[str, str], Tuple[str, int]]]]
        ] = dict()
        self.module_device: Dict[str, Union[int, str, torch.device]] = {}
        self.task_weights: Dict[str, float] = dict()
        self.require_prob_for_evals: Dict[str, bool] = dict()
        self.require_pred_for_evals: Dict[str, bool] = dict()

        # Build network with given tasks
        if tasks is not None:
            self.add_tasks(tasks)

        if Meta.config["meta_config"]["verbose"]:
            logger.info(
                f"Created emmental model {self.name} that contains "
                f"task {self.task_names}."
            )

    def _get_default_device(self) -> torch.device:
        return (
            torch.device("cpu")
            if Meta.config["model_config"]["device"] == -1
            else torch.device(Meta.config["model_config"]["device"])
        )

    def _move_to_device(self) -> None:
        """Move model to specified device."""
        default_device = self._get_default_device()

        for module_name in self.module_pool.keys():
            device = (
                self.module_device[module_name]
                if module_name in self.module_device
                else default_device
            )
            if device != torch.device("cpu"):
                if torch.cuda.is_available():
                    if Meta.config["meta_config"]["verbose"]:
                        logger.info(f"Moving {module_name} module to GPU ({device}).")
                    self.module_pool[module_name].to(device)
                else:
                    if Meta.config["meta_config"]["verbose"]:
                        logger.info(
                            f"No cuda device available. "
                            f"Switch {module_name} to cpu instead."
                        )
                    self.module_pool[module_name].to(torch.device("cpu"))
            else:
                if Meta.config["meta_config"]["verbose"]:
                    logger.info(f"Moving {module_name} module to CPU.")
                self.module_pool[module_name].to(torch.device("cpu"))

    def _to_dataparallel(self) -> None:
        default_device = self._get_default_device()

        for module_name in self.module_pool.keys():
            device = (
                self.module_device[module_name]
                if module_name in self.module_device
                else default_device
            )
            if device != torch.device("cpu"):
                self.module_pool[module_name] = torch.nn.DataParallel(
                    self.module_pool[module_name]
                )

    def _to_distributed_dataparallel(self) -> None:
        # TODO support multiple device with DistributedDataParallel
        for key in self.module_pool.keys():
            # Ensure there is some gradient parameter for DDP
            if not any(p.requires_grad for p in self.module_pool[key].parameters()):
                continue
            self.module_pool[
                key
            ] = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                self.module_pool[key],
                device_ids=[Meta.config["learner_config"]["local_rank"]],
                output_device=Meta.config["learner_config"]["local_rank"],
                find_unused_parameters=True,
            )

    def add_tasks(self, tasks: Union[EmmentalTask, List[EmmentalTask]]) -> None:
        """Build the MTL network using all tasks.

        Args:
          tasks: A task or a list of tasks.
        """
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        for task in tasks:
            self.add_task(task)

    def add_task(self, task: EmmentalTask) -> None:
        """Add a single task into MTL network.

        Args:
          task: A task to add.
        """
        if not isinstance(task, EmmentalTask):
            raise ValueError(f"Unrecognized task type {task}.")

        if task.name in self.task_names:
            raise ValueError(
                f"Found duplicate task {task.name}, different task should use "
                f"different task name."
            )

        # Combine module_pool from all tasks
        for key in task.module_pool.keys():
            if key in self.module_pool.keys():
                task.module_pool[key] = self.module_pool[key]
            else:
                self.module_pool[key] = task.module_pool[key]
        # Collect task name
        self.task_names.add(task.name)
        # Collect task flow
        self.task_flows[task.name] = task.task_flow
        # Collect loss function
        self.loss_funcs[task.name] = task.loss_func
        # Collect output function
        self.output_funcs[task.name] = task.output_func
        # Collect action outputs
        self.action_outputs[task.name] = task.action_outputs
        # Collect module device
        self.module_device.update(task.module_device)
        # Collect scorer
        self.scorers[task.name] = task.scorer
        # Collect weight
        self.task_weights[task.name] = task.weight
        # Collect require prob for eval
        self.require_prob_for_evals[task.name] = task.require_prob_for_eval
        # Collect require pred for eval
        self.require_pred_for_evals[task.name] = task.require_pred_for_eval

        # Move model to specified device
        self._move_to_device()

    def update_task(self, task: EmmentalTask) -> None:
        """Update a existing task in MTL network.

        Args:
          task: A task to update.
        """
        # Update module_pool with task
        for key in task.module_pool.keys():
            # Update the model's module with the task's module
            self.module_pool[key] = task.module_pool[key]
        # Update task flow
        self.task_flows[task.name] = task.task_flow
        # Update loss function
        self.loss_funcs[task.name] = task.loss_func
        # Update output function
        self.output_funcs[task.name] = task.output_func
        # Update action outputs
        self.action_outputs[task.name] = task.action_outputs
        # Update module device
        self.module_device.update(task.module_device)
        # Update scorer
        self.scorers[task.name] = task.scorer
        # Update weight
        self.task_weights[task.name] = task.weight
        # Update require prob for eval
        self.require_prob_for_evals[task.name] = task.require_prob_for_eval
        # Update require pred for eval
        self.require_pred_for_evals[task.name] = task.require_pred_for_eval

        # Move model to specified device
        self._move_to_device()

    def remove_task(self, task_name: str) -> None:
        """Remove a existing task from MTL network.

        Args:
          task_name: The task name to remove.
        """
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
        del self.action_outputs[task_name]
        del self.scorers[task_name]
        del self.task_weights[task_name]
        del self.require_prob_for_evals[task_name]
        del self.require_pred_for_evals[task_name]
        # TODO: remove the modules only associate with that task

    def __repr__(self) -> str:
        """Represent the model as a string."""
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

    def flow(self, X_dict: Dict[str, Any], task_names: List[str]) -> Dict[str, Any]:
        """Forward based on input and task flow.

        Note:
          We assume that all shared modules from all tasks are based on the
          same input.

        Args:
          X_dict: The input data
          task_names: The task names that needs to forward.

        Returns:
          The output of all forwarded modules
        """
        default_device = self._get_default_device()

        X_dict = move_to_device(X_dict, default_device)

        output_dict = dict(_input_=X_dict)

        # Call forward for each task
        for task_name in task_names:
            for action in self.task_flows[task_name]:
                if action["name"] not in output_dict:
                    if action["inputs"]:
                        try:
                            action_module_device = (
                                self.module_device[action["module"]]
                                if action["module"] in self.module_device
                                else default_device
                            )
                            input = move_to_device(
                                [
                                    output_dict[action_name][output_index]
                                    for action_name, output_index in action["inputs"]
                                ],
                                action_module_device,
                            )
                        except Exception:
                            raise ValueError(f"Unrecognized action {action}.")
                        output = self.module_pool[action["module"]].forward(*input)
                    else:
                        # TODO: Handle multiple device with not inputs case
                        output = self.module_pool[action["module"]].forward(output_dict)
                    if isinstance(output, tuple):
                        output = list(output)
                    if not isinstance(output, list) and not isinstance(output, dict):
                        output = [output]
                    output_dict[action["name"]] = output

        return output_dict

    def forward(  # type: ignore
        self,
        uids: List[str],
        X_dict: Dict[str, Any],
        Y_dict: Dict[str, Tensor],
        task_to_label_dict: Dict[str, str],
        return_loss=True,
        return_probs=True,
        return_action_outputs=False,
    ) -> Union[
        Tuple[
            Dict[str, List[str]],
            Dict[str, Tensor],
            Dict[str, Union[ndarray, List[ndarray]]],
            Dict[str, Union[ndarray, List[ndarray]]],
            Dict[str, Dict[str, Union[ndarray, List]]],
        ],
        Tuple[
            Dict[str, List[str]],
            Dict[str, Tensor],
            Dict[str, Union[ndarray, List[ndarray]]],
            Dict[str, Union[ndarray, List[ndarray]]],
        ],
    ]:
        """Forward function.

        Args:
          uids: The uids of input data.
          X_dict: The input data.
          Y_dict: The output data.
          task_to_label_dict: The task to label mapping.
          return_loss: Whether return loss or not, defaults to True.
          return_probs: Whether return probs or not, defaults to True.
          return_action_outputs: Whether return action_outputs or not,
          defaults to False.

        Returns:
          The (active) uids, loss, prob, gold, action_output (optional) in the batch of
          all tasks.
        """
        uid_dict: Dict[str, List[str]] = defaultdict(list)
        loss_dict: Dict[str, Tensor] = defaultdict(Tensor) if return_loss else None
        gold_dict: Dict[str, Union[ndarray, List[ndarray]]] = (
            defaultdict(list) if Y_dict is not None else None
        )
        prob_dict: Dict[str, Union[ndarray, List[ndarray]]] = (
            defaultdict(list) if return_probs else None
        )
        out_dict: Dict[str, Dict[str, Union[ndarray, List]]] = (
            defaultdict(lambda: defaultdict(list)) if return_action_outputs else None
        )

        task_names = list(task_to_label_dict.keys())

        output_dict = self.flow(X_dict, task_names)

        if Y_dict is not None:
            # Calculate logits and loss for each task
            for task_name, label_name in task_to_label_dict.items():
                Y = Y_dict[label_name] if label_name is not None else None

                # Select the active samples
                if Y is not None:
                    if Meta.config["learner_config"]["ignore_index"] is not None:
                        if len(Y.size()) == 1:
                            active = (
                                Y.detach()
                                != Meta.config["learner_config"]["ignore_index"]
                            )
                        else:
                            active = torch.any(
                                Y.detach()
                                != Meta.config["learner_config"]["ignore_index"],
                                dim=1,
                            )
                    else:
                        active = torch.BoolTensor([True] * Y.size()[0])  # type: ignore
                else:
                    active = None

                # Only calculate the loss when active example exists
                if active is None or active.any():
                    if active is not None:
                        uid_dict[task_name] = [
                            *itertools.compress(uids, active.numpy())
                        ]
                    else:
                        uid_dict[task_name] = uids
                    if (
                        return_loss
                        and task_name in self.loss_funcs
                        and self.loss_funcs[task_name] is not None
                    ):
                        loss_dict[task_name] = self.loss_funcs[task_name](
                            output_dict,
                            move_to_device(
                                Y_dict[label_name],
                                Meta.config["model_config"]["device"],
                            )
                            if label_name is not None
                            else None,
                            move_to_device(
                                active, Meta.config["model_config"]["device"]
                            )
                            if active is not None
                            else None,
                        )

                    if (
                        return_probs
                        and task_name in self.output_funcs
                        and self.output_funcs[task_name] is not None
                    ):
                        if active is not None:
                            prob_dict[task_name] = (
                                self.output_funcs[task_name](output_dict)[
                                    move_to_device(
                                        active, Meta.config["model_config"]["device"]
                                    )
                                ]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        else:
                            prob_dict[task_name] = (
                                self.output_funcs[task_name](output_dict)
                                .cpu()
                                .detach()
                                .numpy()
                            )

                    if active is not None:
                        gold_dict[task_name] = Y_dict[label_name][active].cpu().numpy()

                    if (
                        return_action_outputs
                        and task_name in self.action_outputs
                        and self.action_outputs[task_name] is not None
                    ):
                        for action_name, output_index in self.action_outputs[task_name]:
                            if active is not None:
                                out_dict[task_name][f"{action_name}_{output_index}"] = (
                                    output_dict[action_name][output_index][
                                        move_to_device(
                                            active,
                                            Meta.config["model_config"]["device"],
                                        )
                                    ]
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                            else:
                                out_dict[task_name][f"{action_name}_{output_index}"] = (
                                    output_dict[action_name][output_index]
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )

        else:
            # Calculate logits for each task
            for task_name, label_name in task_to_label_dict.items():
                assert (
                    label_name is None
                ), f"Task {task_name} has not {label_name} label."

                uid_dict[task_name] = uids
                if (
                    return_loss
                    and task_name in self.loss_funcs
                    and self.loss_funcs[task_name] is not None
                ):
                    loss_dict[task_name] = self.loss_funcs[task_name](
                        output_dict,
                        None,
                        move_to_device(
                            torch.BoolTensor([True] * len(uids)),
                            Meta.config["model_config"]["device"],
                        ),
                    )

                if (
                    return_probs
                    and task_name in self.output_funcs
                    and self.output_funcs[task_name] is not None
                ):
                    prob_dict[task_name] = (
                        self.output_funcs[task_name](output_dict).cpu().detach().numpy()
                    )

                if (
                    return_action_outputs
                    and task_name in self.action_outputs
                    and self.action_outputs[task_name] is not None
                ):
                    for action_name, output_index in self.action_outputs[task_name]:
                        out_dict[task_name][f"{action_name}_{output_index}"] = (
                            output_dict[action_name][output_index]
                            .cpu()
                            .detach()
                            .numpy()
                        )

        if return_action_outputs:
            return uid_dict, loss_dict, prob_dict, gold_dict, out_dict
        else:
            return uid_dict, loss_dict, prob_dict, gold_dict

    @torch.no_grad()
    def predict(
        self,
        dataloader: EmmentalDataLoader,
        return_loss: bool = True,
        return_probs: bool = True,
        return_preds: bool = False,
        return_action_outputs: bool = False,
    ) -> Dict[str, Any]:
        """Predict from dataloader.

        Args:
          dataloader: The dataloader to predict.
          return_loss: Whether return loss or not, defaults to True.
          return_probs: Whether return probs or not, defaults to True.
          return_preds: Whether return predictions or not, defaults to False.
          return_action_outputs: Whether return action_outputs or not,
          defaults to False.

        Returns:
          The result dict.
        """
        self.eval()

        # Check if Y_dict exists
        has_y_dict = False if isinstance(dataloader.dataset[0], dict) else True

        uid_dict: Dict[str, List[str]] = defaultdict(list)
        prob_dict: Dict[str, Union[ndarray, List[ndarray]]] = (
            defaultdict(list) if return_probs else None
        )
        pred_dict: Dict[str, Union[ndarray, List[ndarray]]] = (
            defaultdict(list) if return_preds else None
        )
        out_dict: Dict[str, Dict[str, List[Union[ndarray, int, float]]]] = (
            defaultdict(lambda: defaultdict(list)) if return_action_outputs else None
        )
        loss_dict: Dict[str, Union[ndarray, float]] = (
            defaultdict(list) if return_loss else None  # type: ignore
        )
        gold_dict: Dict[str, List[Union[ndarray, int, float]]] = (
            defaultdict(list) if has_y_dict else None
        )

        with torch.no_grad():
            for bdict in tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Evaluating {dataloader.data_name} ({dataloader.split})",
            ):
                if has_y_dict:
                    X_bdict, Y_bdict = bdict
                else:
                    X_bdict = bdict
                    Y_bdict = None

                if return_action_outputs:
                    (
                        uid_bdict,
                        loss_bdict,
                        prob_bdict,
                        gold_bdict,
                        out_bdict,
                    ) = self.forward(  # type: ignore
                        X_bdict[dataloader.uid],
                        X_bdict,
                        Y_bdict,
                        dataloader.task_to_label_dict,
                        return_loss=return_loss,
                        return_action_outputs=return_action_outputs,
                        return_probs=return_probs or return_preds,
                    )
                else:
                    (
                        uid_bdict,
                        loss_bdict,
                        prob_bdict,
                        gold_bdict,
                    ) = self.forward(  # type: ignore
                        X_bdict[dataloader.uid],
                        X_bdict,
                        Y_bdict,
                        dataloader.task_to_label_dict,
                        return_loss=return_loss,
                        return_action_outputs=return_action_outputs,
                        return_probs=return_probs or return_preds,
                    )
                    out_bdict = None
                for task_name in uid_bdict.keys():
                    uid_dict[task_name].extend(uid_bdict[task_name])
                    if return_loss:
                        if len(loss_bdict[task_name].size()) == 0:
                            if loss_dict[task_name] == []:
                                loss_dict[task_name] = 0
                            loss_dict[task_name] += loss_bdict[task_name].item() * len(
                                uid_bdict[task_name]
                            )
                        else:
                            loss_dict[task_name].extend(  # type: ignore
                                loss_bdict[task_name].cpu().numpy()
                            )
                    if return_probs:
                        prob_dict[task_name].extend(  # type: ignore
                            prob_bdict[task_name]
                        )
                    if return_preds:
                        pred_dict[task_name].extend(  # type: ignore
                            prob_to_pred(prob_bdict[task_name])
                        )
                    if has_y_dict:
                        gold_dict[task_name].extend(gold_bdict[task_name])
                if return_action_outputs and out_bdict:
                    for task_name in out_bdict.keys():
                        for action_name in out_bdict[task_name].keys():
                            out_dict[task_name][action_name].extend(
                                out_bdict[task_name][action_name]
                            )

        # Calculate average loss
        if return_loss:
            for task_name in uid_dict.keys():
                if not isinstance(loss_dict[task_name], list):
                    loss_dict[task_name] /= len(uid_dict[task_name])

        res = {
            "uids": uid_dict,
            "golds": gold_dict,
            "losses": loss_dict,
        }

        if return_probs:
            for task_name in prob_dict.keys():
                prob_dict[task_name] = array_to_numpy(prob_dict[task_name])
            res["probs"] = prob_dict

        if return_preds:
            for task_name in pred_dict.keys():
                pred_dict[task_name] = array_to_numpy(pred_dict[task_name])
            res["preds"] = pred_dict

        if return_action_outputs:
            res["outputs"] = out_dict

        return res

    @torch.no_grad()
    def score(
        self,
        dataloaders: Union[EmmentalDataLoader, List[EmmentalDataLoader]],
        return_average: bool = True,
    ) -> Dict[str, float]:
        """Score the data from dataloader.

        Args:
          dataloaders: The dataloaders to score.
          return_average: Whether to return average score.

        Returns:
          Score dict.
        """
        self.eval()

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        metric_score_dict = dict()

        if return_average:
            micro_score_dict: defaultdict = defaultdict(list)
            macro_score_dict: defaultdict = defaultdict(list)
            macro_loss_dict: defaultdict = defaultdict(list)

        for dataloader in dataloaders:
            return_probs = False
            return_preds = False
            has_scorer = False

            for task_name in dataloader.task_to_label_dict:
                return_probs = return_probs or self.require_prob_for_evals[task_name]
                return_preds = return_preds or self.require_pred_for_evals[task_name]
                if self.scorers[task_name]:
                    has_scorer = True

            if not has_scorer:
                logger.warning(
                    f"Task(s) used in dataloader {dataloader.data_name} do(es)n't "
                    "have scorer, continue..."
                )
                continue

            predictions = self.predict(
                dataloader,
                return_probs=return_probs,
                return_preds=return_preds,
                return_action_outputs=False,
            )
            for task_name in predictions["uids"].keys():
                metric_score = self.scorers[task_name].score(
                    predictions["golds"][task_name],
                    predictions["probs"][task_name] if return_probs else None,
                    predictions["preds"][task_name] if return_preds else None,
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
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    predictions["losses"][task_name]
                )

                if return_average:
                    # Collect average score
                    identifier = construct_identifier(
                        task_name, dataloader.data_name, dataloader.split, "average"
                    )
                    metric_score_dict[identifier] = np.mean(  # type: ignore
                        list(metric_score.values())
                    )

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
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    micro_score_dict[split]
                )
                identifier = construct_identifier(
                    "model", "all", split, "macro_average"
                )
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    macro_score_dict[split]
                )
                identifier = construct_identifier("model", "all", split, "loss")
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    macro_loss_dict[split]
                )

            # Collect overall micro/macro average score/loss
            if len(micro_score_dict):
                identifier = construct_identifier(
                    "model", "all", "all", "micro_average"
                )
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    list(itertools.chain.from_iterable(micro_score_dict.values()))
                )
            if len(macro_score_dict):
                identifier = construct_identifier(
                    "model", "all", "all", "macro_average"
                )
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    list(itertools.chain.from_iterable(macro_score_dict.values()))
                )
            if len(macro_loss_dict):
                identifier = construct_identifier("model", "all", "all", "loss")
                metric_score_dict[identifier] = np.mean(  # type: ignore
                    list(itertools.chain.from_iterable(macro_loss_dict.values()))
                )

        # TODO: have a better to handle global evaluation metric
        if Meta.config["learner_config"]["global_evaluation_metric_dict"]:
            global_evaluation_metric_dict = Meta.config["learner_config"][
                "global_evaluation_metric_dict"
            ]
            for metric_name, metric in global_evaluation_metric_dict.items():
                metric_score_dict[metric_name] = metric(metric_score_dict)

        return metric_score_dict

    def save(
        self,
        model_path: str,
        iteration: Optional[Union[float, int]] = None,
        metric_dict: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> None:
        """Save model.

        Args:
          model_path: Saved model path.
          iteration: The iteration of the model, defaults to `None`.
          metric_dict: The metric dict, defaults to `None`.
          verbose: Whether log the info, defaults to `True`.
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
            },
            "iteration": iteration,
            "metric_dict": metric_dict,
        }

        try:
            torch.save(state_dict, model_path)
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

        if Meta.config["meta_config"]["verbose"] and verbose:
            logger.info(f"[{self.name}] Model saved in {model_path}")

    def load(
        self,
        model_path: str,
        verbose: bool = True,
    ) -> None:
        """Load model state_dict from file and reinitialize the model weights.

        Args:
          model_path: Saved model path.
          verbose: Whether log the info, defaults to `True`.
        """
        if not os.path.exists(model_path):
            logger.error("Loading failed... Model does not exist.")

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        except BaseException:
            logger.error(f"Loading failed... Cannot load model from {model_path}")
            raise

        self.load_state_dict(checkpoint["model"]["module_pool"])

        if Meta.config["meta_config"]["verbose"] and verbose:
            logger.info(f"[{self.name}] Model loaded from {model_path}")

        # Move model to specified device
        self._move_to_device()

    def collect_state_dict(self) -> Dict[str, Any]:
        """Collect the state dict."""
        state_dict: Dict[str, Any] = defaultdict(list)

        for module_name, module in self.module_pool.items():
            if hasattr(module, "module"):
                state_dict[module_name] = module.module.state_dict()  # type: ignore
            else:
                state_dict[module_name] = module.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore
        """Load the state dict.

        Args:
          state_dict: The state dict to load.
        """
        for module_name, module_state_dict in state_dict.items():
            if module_name in self.module_pool:
                if hasattr(self.module_pool[module_name], "module"):
                    self.module_pool[module_name].module.load_state_dict(
                        module_state_dict
                    )
                else:
                    self.module_pool[module_name].load_state_dict(module_state_dict)
            else:
                logger.info(f"Missing {module_name} in module_pool, skip it..")
