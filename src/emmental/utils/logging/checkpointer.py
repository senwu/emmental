import logging
import os
from shutil import copyfile

import torch

from emmental.meta import Meta

logger = logging.getLogger(__name__)


class Checkpointer(object):
    """Checkpointing Training logging class to log train infomation"""

    def __init__(self):
        # Set up checkpoint directory
        self.checkpoint_path = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_path"
        ]
        if self.checkpoint_path is None:
            self.checkpoint_path = Meta.log_path

        # Create checkpoint directory if necessary
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.checkpoint_freq = int(
            Meta.config["logging_config"]["evaluation_freq"]
            * Meta.config["logging_config"]["checkpointer_config"]["checkpoint_freq"]
        )

        if self.checkpoint_freq <= 0:
            raise ValueError(
                f"Invalid checkpoint freq {self.checkpoint_freq}, "
                f"must be greater 0."
            )

        self.checkpoint_unit = Meta.config["logging_config"]["counter_unit"]

        logger.info(
            f"Save checkpoints at {self.checkpoint_path} every "
            f"{self.checkpoint_freq} {self.checkpoint_unit}"
        )

        self.checkpoint_metric = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_metric"
        ]
        if not isinstance(self.checkpoint_metric, list):
            self.checkpoint_metric = [self.checkpoint_metric]

        self.checkpoint_metric_mode = Meta.config["logging_config"][
            "checkpointer_config"
        ]["checkpoint_metric_mode"].lower()

        if self.checkpoint_metric_mode not in ["min", "max"]:
            raise ValueError(
                f"Unrecognized checkpoint metric mode {self.checkpoint_metric_mode}, "
                f"must be 'min' or 'max'."
            )

        self.checkpoint_runway = Meta.config["logging_config"]["checkpointer_config"][
            "checkpoint_runway"
        ]
        logger.info(
            f"No checkpoints saved before {self.checkpoint_runway} "
            f"{self.checkpoint_unit}."
        )

        self.best_metric_dict = dict()

    def checkpoint(self, iteration, model, optimizer, lr_scheduler, metric_dict):
        # Check the checkpoint_runway condition is met
        if iteration < self.checkpoint_runway:
            return
        elif iteration == self.checkpoint_runway:
            logger.info(
                f"checkpoint_runway condition has been met. Start checkpoining."
            )

        if iteration > 0 and iteration % self.checkpoint_freq == 0:
            state_dict = self.collect_state_dict(
                iteration, model, optimizer, lr_scheduler, metric_dict
            )
            checkpoint_path = f"{self.checkpoint_path}/checkpoint_{iteration}.pth"
            torch.save(state_dict, checkpoint_path)
            logger.info(
                f"Save checkpoint of {iteration} {self.checkpoint_unit} "
                f"at {checkpoint_path}."
            )

            if not set(self.checkpoint_metric).isdisjoint(metric_dict):
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

    def is_new_best(self, metric_dict):
        best_metric = set()

        for metric in self.checkpoint_metric:
            if metric not in self.best_metric_dict:
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_metric_mode == "max"
                and metric_dict[metric] > self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)
            elif (
                self.checkpoint_metric_mode == "min"
                and metric_dict[metric] < self.best_metric_dict[metric]
            ):
                self.best_metric_dict[metric] = metric_dict[metric]
                best_metric.add(metric)

        return best_metric

    def collect_state_dict(
        self, iteration, model, optimizer, lr_scheduler, metric_dict
    ):
        """Generate the state dict of the model."""
        model_params = {
            "name": model.name,
            "module_pool": model.module_pool,
            "task_flows": model.task_flows,
            "loss_funcs": model.loss_funcs,
            "output_funcs": model.output_funcs,
        }
        state_dict = {
            "iteration": iteration,
            "model": model_params,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "metric_dict": metric_dict,
        }

        return state_dict

    def load_best_model(self, model):
        """Load the best model from the checkpoint."""
        if not bool(self.best_metric_dict):
            logger.info(f"No best model found.")
        else:
            # TODO: only load the best model of the first metric in best_metric_dict
            metric = self.best_metric_dict.keys()[0]
            state_dict = torch.load(
                f"{self.checkpoint_path}/best_model_{metric.replace('/', '_')}.pth",
                map_location=torch.device("cpu"),
            )
        model.name = state_dict["model"]["name"]
        model.module_pool = state_dict["model"]["module_pool"]
        model.task_flows = state_dict["model"]["task_flows"]
        model.loss_funcs = state_dict["model"]["loss_funcs"]
        model.output_funcs = state_dict["model"]["output_funcs"]

        return model
