"""Emmental logging manager."""
import logging
from typing import Dict, Union

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from emmental.logging.checkpointer import Checkpointer
from emmental.logging.json_writer import JsonWriter
from emmental.logging.log_writer import LogWriter
from emmental.logging.tensorboard_writer import TensorBoardWriter
from emmental.logging.wandb_writer import WandbWriter
from emmental.meta import Meta
from emmental.model import EmmentalModel

logger = logging.getLogger(__name__)


class LoggingManager(object):
    """A class to manage logging during training progress.

    Args:
      n_batches_per_epoch: Total number batches per epoch.
    """

    def __init__(
        self, n_batches_per_epoch: int, epoch_count: int = 0, batch_count: int = 0
    ) -> None:
        """Initialize LoggingManager."""
        self.n_batches_per_epoch = n_batches_per_epoch

        # Set up counter

        # Set up evaluation/checkpointing unit (sample, batch, epoch)
        self.counter_unit = Meta.config["logging_config"]["counter_unit"]

        if self.counter_unit not in ["sample", "batch", "epoch"]:
            raise ValueError(f"Unrecognized unit: {self.counter_unit}")

        # Set up evaluation frequency
        self.evaluation_freq = Meta.config["logging_config"]["evaluation_freq"]
        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"Evaluating every {self.evaluation_freq} {self.counter_unit}.")

        if Meta.config["logging_config"]["checkpointing"]:
            self.checkpointing = True

            # Set up checkpointing frequency
            self.checkpointing_freq = int(
                Meta.config["logging_config"]["checkpointer_config"]["checkpoint_freq"]
            )
            if Meta.config["meta_config"]["verbose"]:
                logger.info(
                    f"Checkpointing every "
                    f"{self.checkpointing_freq * self.evaluation_freq} "
                    f"{self.counter_unit}."
                )

            # Set up checkpointer
            self.checkpointer = Checkpointer()
        else:
            self.checkpointing = False
            if Meta.config["meta_config"]["verbose"]:
                logger.info("No checkpointing.")

        # Set up number of samples passed since last evaluation/checkpointing and
        # total number of samples passed since learning process
        self.sample_count: int = 0
        self.sample_total: int = 0

        # Set up number of batches passed since last evaluation/checkpointing and
        # total number of batches passed since learning process
        self.batch_count: int = batch_count
        self.batch_total: int = batch_count
        if self.batch_count != 0:
            if self.counter_unit == "batch":
                while self.batch_count >= self.evaluation_freq:
                    self.batch_count -= self.evaluation_freq
            elif self.counter_unit == "epoch":
                while (
                    self.batch_count >= self.evaluation_freq * self.n_batches_per_epoch
                ):
                    self.batch_count -= self.evaluation_freq * self.n_batches_per_epoch

        # Set up number of epochs passed since last evaluation/checkpointing and
        # total number of epochs passed since learning process
        self.epoch_count: Union[float, int] = epoch_count
        self.epoch_total: Union[float, int] = epoch_count
        if self.epoch_count != 0:
            if self.counter_unit == "epoch":
                while self.epoch_count >= self.evaluation_freq:
                    self.epoch_count -= self.evaluation_freq
            elif self.counter_unit == "batch":
                while (
                    self.epoch_count >= self.evaluation_freq / self.n_batches_per_epoch
                ):
                    self.epoch_count -= self.evaluation_freq / self.n_batches_per_epoch

        # Set up number of unit passed since last evaluation/checkpointing and
        # total number of unit passed since learning process
        self.unit_count: Union[float, int] = 0
        self.unit_total: Union[float, int] = 0

        # Set up count that triggers the evaluation since last checkpointing
        self.trigger_count = 0

        # Set up log writer
        writer_opt = Meta.config["logging_config"]["writer_config"]["writer"]

        if writer_opt is None:
            self.writer = LogWriter()
        elif writer_opt == "json":
            self.writer = JsonWriter()
        elif writer_opt == "tensorboard":
            self.writer = TensorBoardWriter()
        elif writer_opt == "wandb":
            self.writer = WandbWriter()
        else:
            raise ValueError(f"Unrecognized writer option '{writer_opt}'")

        self.log_unit_sanity_check = False

    def update(self, batch_size: int) -> None:
        """Update the counter.

        Args:
          batch_size: The number of the samples in the batch.
        """
        # Update number of samples
        self.sample_count += batch_size
        self.sample_total += batch_size

        # Update number of batches
        self.batch_count += 1
        self.batch_total += 1

        # Update number of epochs
        self.epoch_count = self.batch_count / self.n_batches_per_epoch
        self.epoch_total = self.batch_total / self.n_batches_per_epoch
        if self.epoch_count == int(self.epoch_count):
            self.epoch_count = int(self.epoch_count)
        if self.epoch_total == int(self.epoch_total):
            self.epoch_total = int(self.epoch_total)

        # Update number of units
        if self.counter_unit == "sample":
            self.unit_count = self.sample_count
            self.unit_total = self.sample_total
        if self.counter_unit == "batch":
            self.unit_count = self.batch_count
            self.unit_total = self.batch_total
        elif self.counter_unit == "epoch":
            self.unit_count = self.epoch_count
            self.unit_total = self.epoch_total

    def trigger_evaluation(self) -> bool:
        """Check if triggers the evaluation."""
        satisfied = self.unit_count >= self.evaluation_freq
        if satisfied:
            self.trigger_count += 1
            self.reset()
        return satisfied

    def trigger_checkpointing(self) -> bool:
        """Check if triggers the checkpointing."""
        if not self.checkpointing:
            return False
        satisfied = self.trigger_count >= self.checkpointing_freq
        if satisfied:
            self.trigger_count = 0
        return satisfied

    def reset(self) -> None:
        """Reset the counter."""
        self.sample_count = 0
        self.batch_count = 0
        self.epoch_count = 0
        self.unit_count = 0

    def write_log(self, metric_dict: Dict[str, float]) -> None:
        """Write the metrics to the log.

        Args:
          metric_dict: The metric dict.
        """
        unit_total: Union[float, int] = self.unit_total
        # As Tensorboard/Wandb only allow integer values for unit count, emmental casts
        # non integer value to integer and switch the counter unit from epoch to batch.
        if (
            Meta.config["logging_config"]["writer_config"]["writer"]
            in ["tensorboard", "wandb"]
            and self.counter_unit == "epoch"
            and int(self.evaluation_freq) != self.evaluation_freq
        ):
            if not self.log_unit_sanity_check:
                logger.warning(
                    "Cannot use float value for evaluation_freq when counter_unit "
                    "uses epoch with tensorboard writer, switch to batch as "
                    "count_unit."
                )
                self.log_unit_sanity_check = True
            unit_total = self.batch_total
        self.writer.add_scalar_dict(metric_dict, unit_total)

    def checkpoint_model(
        self,
        model: EmmentalModel,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        metric_dict: Dict[str, float],
    ) -> None:
        """Checkpoint the model.

        Args:
          model: The model to checkpoint.
          optimizer: The optimizer used during training process.
          lr_scheduler: Learning rate scheduler.
          metric_dict: the metric dict.
        """
        self.checkpointer.checkpoint(
            self.unit_total, model, optimizer, lr_scheduler, metric_dict
        )

    def close(self, model: EmmentalModel) -> EmmentalModel:
        """Close the checkpointer and reload the model if necessary.

        Args:
          model: The trained model.

        Returns:
          The reloaded model if necessary
        """
        self.writer.close()
        if self.checkpointing:
            model = self.checkpointer.load_best_model(model)
            self.checkpointer.clear()
        return model
