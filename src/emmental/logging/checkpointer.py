import glob
import logging
import os
from shutil import copyfile
from typing import Any, Dict, List, Set, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from emmental.meta import Meta
from emmental.model import EmmentalModel

logger = logging.getLogger(__name__)


class Checkpointer(object):
    r"""Checkpointing class to log train information"""

    def __init__(self) -> None:
        r"""Initialize the checkpointer.
        """

        # Set up checkpoint directory
        self.checkpoint_path = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_path"
        ]
        if self.checkpoint_path is None:
            self.checkpoint_path = Meta.log_path

        # Create checkpoint directory if necessary
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # Set up checkpoint frequency
        self.checkpoint_freq = (
            Meta.config["logging_config"]["evaluation_freq"]
            * Meta.config["logging_config"]["checkpointer_config"]["checkpoint_freq"]
        )

        if self.checkpoint_freq <= 0:
            raise ValueError(
                f"Invalid checkpoint freq {self.checkpoint_freq}, "
                f"must be greater 0."
            )

        # Set up checkpoint unit
        self.checkpoint_unit = Meta.config["logging_config"]["counter_unit"]

        logger.info(
            f"Save checkpoints at {self.checkpoint_path} every "
            f"{self.checkpoint_freq} {self.checkpoint_unit}"
        )

        # Set up checkpoint metric
        self.checkpoint_metric = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_metric"
        ]

        self.checkpoint_all_metrics = Meta.config["logging_config"][
            "checkpointer_config"
        ]["checkpoint_task_metrics"]

        # Collect all metrics to checkpoint
        if self.checkpoint_all_metrics is None:
            self.checkpoint_all_metrics = dict()

        if self.checkpoint_metric:
            self.checkpoint_all_metrics.update(self.checkpoint_metric)

        # Check evaluation metric mode
        for metric, mode in self.checkpoint_all_metrics.items():
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"Unrecognized checkpoint metric mode {mode} for metric {metric}, "
                    f"must be 'min' or 'max'."
                )

        self.checkpoint_runway = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_runway"
        ]
        logger.info(
            f"No checkpoints saved before {self.checkpoint_runway} "
            f"{self.checkpoint_unit}."
        )

        self.checkpoint_all = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_all"
        ]
        logger.info(f"Checkpointing all checkpoints: {self.checkpoint_all}.")

        self.checkpoint_paths: List[str] = []

        # Set up checkpoint clear
        self.clear_intermediate_checkpoints = Meta.config["logging_config"][
            "checkpointer_config"
        ]["clear_intermediate_checkpoints"]
        self.clear_all_checkpoints = Meta.config["logging_config"][
            "checkpointer_config"
        ]["clear_all_checkpoints"]

        # Set up checkpoint flag
        self.checkpoint_condition_met = False

        self.best_metric_dict: Dict[str, float] = dict()

    def checkpoint(
        self,
        iteration: Union[float, int],
        model: EmmentalModel,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        metric_dict: Dict[str, float],
    ) -> None:
        r"""Checkpointing the checkpoint.

        Args:
          iteration(float or int): The current iteration.
          model(EmmentalModel): The model to checkpoint.
          optimizer(Optimizer): The optimizer used during training process.
          lr_scheduler(_LRScheduler): Learning rate scheduler.
          metric_dict(dict): The metric dict.

        """

        # Check the checkpoint_runway condition is met
        if iteration < self.checkpoint_runway:
            return
        elif not self.checkpoint_condition_met and iteration >= self.checkpoint_runway:
            self.checkpoint_condition_met = True
            logger.info(
                f"checkpoint_runway condition has been met. Start checkpoining."
            )

        state_dict = self.collect_state_dict(
            iteration, model, optimizer, lr_scheduler, metric_dict
        )
        checkpoint_path = f"{self.checkpoint_path}/checkpoint_{iteration}.pth"
        torch.save(state_dict, checkpoint_path)
        logger.info(
            f"Save checkpoint of {iteration} {self.checkpoint_unit} "
            f"at {checkpoint_path}."
        )

        if self.checkpoint_all is False:
            for path in self.checkpoint_paths:
                if os.path.exists(path):
                    os.remove(path)

        self.checkpoint_paths.append(checkpoint_path)

        if not set(self.checkpoint_all_metrics.keys()).isdisjoint(
            set(metric_dict.keys())
        ):
            new_best_metrics = self.is_new_best(metric_dict)
            for metric in new_best_metrics:
                copyfile(
                    checkpoint_path,
                    f"{self.checkpoint_path}/best_model_"
                    f"{metric.replace('/', '_')}.pth",
                )

                logger.info(
                    f"Save best model of metric {metric} at {self.checkpoint_path}"
                    f"/best_model_{metric.replace('/', '_')}.pth"
                )

    def is_new_best(self, metric_dict: Dict[str, float]) -> Set[str]:
        r"""Update the best score.

        Args:
          metric_dict(dict): The current metric dict.

        Returns:
          set: The updated best metric set.

        """

        best_metric = set()

        for metric in metric_dict:
            if metric not in self.checkpoint_all_metrics:
                continue
            if metric not in self.best_metric_dict:
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_all_metrics[metric] == "max"
                and metric_dict[metric] > self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_all_metrics[metric] == "min"
                and metric_dict[metric] < self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)

        return best_metric

    def clear(self) -> None:
        r"""Clear checkpoints."""

        if self.clear_all_checkpoints:
            logger.info("Clear all checkpoints.")
            file_list = glob.glob(f"{self.checkpoint_path}/*.pth")
            for file in file_list:
                os.remove(file)
        elif self.clear_intermediate_checkpoints:
            logger.info("Clear all intermediate checkpoints.")
            file_list = glob.glob(f"{self.checkpoint_path}/checkpoint_*.pth")
            for file in file_list:
                os.remove(file)

    def collect_state_dict(
        self,
        iteration: Union[float, int],
        model: EmmentalModel,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        metric_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        r"""Collect the state dict of the model.

        Args:
          iteration(float or int): The current iteration.
          model(EmmentalModel): The model to checkpoint.
          optimizer(Optimizer): The optimizer used during training process.
          lr_scheduler(_LRScheduler): Learning rate scheduler.
          metric_dict(dict): the metric dict.

        Returns:
          dict: The state dict.
        """

        model_params = {
            "name": model.name,
            "module_pool": model.collect_state_dict(),
            # "task_names": model.task_names,
            # "task_flows": model.task_flows,
            # "loss_funcs": model.loss_funcs,
            # "output_funcs": model.output_funcs,
            # "scorers": model.scorers,
        }

        state_dict = {
            "iteration": iteration,
            "model": model_params,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "metric_dict": metric_dict,
        }

        return state_dict

    def load_best_model(self, model: EmmentalModel) -> EmmentalModel:
        r"""Load the best model from the checkpoint.

        Args:
          model(EmmentalModel.): The current model.

        Returns:
          EmmentalModel: The best model load from the checkpoint.

        """

        if list(self.checkpoint_metric.keys())[0] not in self.best_metric_dict:
            logger.info(f"No best model found, use the original model.")
        else:
            # Load the best model of checkpoint_metric
            metric = list(self.checkpoint_metric.keys())[0]
            best_model_path = (
                f"{self.checkpoint_path}/best_model_{metric.replace('/', '_')}.pth"
            )
            logger.info(f"Loading the best model from {best_model_path}.")
            checkpoint = torch.load(best_model_path, map_location=torch.device("cpu"))
            model.name = checkpoint["model"]["name"]
            model.load_state_dict(checkpoint["model"]["module_pool"])
            # model.task_names = checkpoint["model"]["task_names"]
            # model.task_flows = checkpoint["model"]["task_flows"]
            # model.loss_funcs = checkpoint["model"]["loss_funcs"]
            # model.output_funcs = checkpoint["model"]["output_funcs"]
            # model.scorers = checkpoint["model"]["scorers"]

            model._move_to_device()

        return model
