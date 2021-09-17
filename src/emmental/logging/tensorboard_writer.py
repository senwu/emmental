"""Emmental tensor board writer."""
import copy
import json
from typing import Union

from torch.utils.tensorboard import SummaryWriter

from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta
from emmental.utils.utils import convert_to_serializable_json


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(self) -> None:
        """Initialize TensorBoardWriter."""
        super().__init__()

        # Set up tensorboard summary writer and save config
        self.writer = SummaryWriter(Meta.log_path)
        self.write_config()

    def add_scalar(
        self, name: str, value: Union[float, int], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          name: The name of the scalar.
          value: The value of the scalar.
          step: The current step.
        """
        self.writer.add_scalar(name, value, step)

    def write_config(self, config_filename: str = "config.yaml") -> None:
        """Write the config to tensorboard and dump it to file.

        Args:
          config_filename: The config filename, defaults to "config.yaml".
        """
        config = json.dumps(convert_to_serializable_json(copy.deepcopy(Meta.config)))
        self.writer.add_text(tag="config", text_string=config)
        super().write_config(config_filename)

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        pass

    def close(self) -> None:
        """Close the tensorboard writer."""
        self.writer.close()
