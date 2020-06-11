"""Emmental log writer."""
import json
import os
from collections import defaultdict
from typing import Any, Dict, Union

import yaml

from emmental.meta import Meta


class LogWriter(object):
    """A class for logging during training process."""

    def __init__(self) -> None:
        """Initialize the log writer."""
        self.run_log: defaultdict = defaultdict(list)

    def add_config(self, config: Dict[str, Any]) -> None:
        """Log config.

        Args:
          config: The current config.
        """
        self.config = config

    def add_scalar(
        self, name: str, value: Union[float, int], step: Union[float, int]
    ) -> None:
        """Log a scalar variable.

        Args:
          name: The name of the scalar.
          value: The value of the scalar.
          step: The current step.
        """
        self.run_log[name].append((step, value))

    def write_config(self, config_filename: str = "config.yaml") -> None:
        """Dump the config to file.

        Args:
          config_filename: The config filename, defaults to "config.yaml".
        """
        config_path = os.path.join(Meta.log_path, config_filename)
        with open(config_path, "w") as yml:
            yaml.dump(self.config, yml, default_flow_style=False, allow_unicode=True)

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        log_path = os.path.join(Meta.log_path, log_filename)
        with open(log_path, "w") as f:
            json.dump(self.run_log, f)

    def close(self) -> None:
        """Close the log writer."""
        pass
