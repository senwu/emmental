"""Emmental log writer."""
import copy
import os
from typing import Dict, Union

import yaml

from emmental.meta import Meta
from emmental.utils.utils import convert_to_serializable_json


class LogWriter(object):
    """A class for logging during training process."""

    def __init__(self) -> None:
        """Initialize the log writer."""
        pass

    def add_scalar_dict(
        self, metric_dict: Dict[str, Union[float, int]], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          metric_dict: The metric dict.
          step: The current step.
        """
        for name, value in metric_dict.items():
            self.add_scalar(name, value, step)

    def add_scalar(
        self, name: str, value: Union[float, int], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          name: The name of the scalar.
          value: The value of the scalar.
          step: The current step.
        """
        pass

    def write_config(self, config_filename: str = "config.yaml") -> None:
        """Dump the config to file.

        Args:
          config_filename: The config filename, defaults to "config.yaml".
        """
        config_path = os.path.join(Meta.log_path, config_filename)
        with open(config_path, "w") as yml:
            yaml.dump(
                convert_to_serializable_json(copy.deepcopy(Meta.config)),
                yml,
                default_flow_style=False,
                allow_unicode=True,
            )

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        pass

    def close(self) -> None:
        """Close the log writer."""
        pass
