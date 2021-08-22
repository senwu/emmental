"""Emmental tensor board writer."""
from typing import Dict, Union

import wandb
from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta


class WandbWriter(LogWriter):
    """A class for logging to wandb during training process."""

    def __init__(self) -> None:
        """Initialize TensorBoardWriter."""
        super().__init__()

        # Set up wandb summary writer
        self.write_config()

    def add_scalar_dict(
        self, metric_dict: Dict[str, Union[float, int]], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          metric_dict: The metric dict.
          step: The current step.
        """
        wandb.log(metric_dict, step=step)

    def write_config(self, config_filename: str = "config.yaml") -> None:
        """Write the config to wandb and dump it to file.

        Args:
          config_filename: The config filename, defaults to "config.yaml".
        """
        wandb.init(
            project=Meta.config["logging_config"]["writer_config"][
                "wandb_project_name"
            ],
            name=Meta.config["logging_config"]["writer_config"]["wandb_run_name"],
            config=Meta.config,
        )

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        pass

    def close(self) -> None:
        """Close the wandb writer."""
        pass
