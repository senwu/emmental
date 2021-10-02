"""Emmental tensor board writer."""
import copy
from typing import Dict, Union

import wandb

from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta
from emmental.utils.utils import convert_to_serializable_json


class WandbWriter(LogWriter):
    """A class for logging to wandb during training process."""

    def __init__(self) -> None:
        """Initialize TensorBoardWriter."""
        super().__init__()

        # Set up wandb summary writer and save config
        wandb.init(
            project=Meta.config["logging_config"]["writer_config"][
                "wandb_project_name"
            ],
            name=Meta.config["logging_config"]["writer_config"]["wandb_run_name"],
            config=convert_to_serializable_json(copy.deepcopy(Meta.config)),
        )

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
