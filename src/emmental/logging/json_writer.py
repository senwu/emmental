# Copyright (c) 2021 Sen Wu. All Rights Reserved.


"""Emmental log writer."""
import json
import os
from collections import defaultdict
from typing import Union

from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta


class JsonWriter(LogWriter):
    """A class for logging during training process."""

    def __init__(self) -> None:
        """Initialize the log writer."""
        super().__init__()

        # Set up json writer
        self.run_log: defaultdict = defaultdict(list)

        # Dump config
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
        self.run_log[name].append((step, value))

    def write_log(self, log_filename: str = "log.json") -> None:
        """Dump the log to file.

        Args:
          log_filename: The log filename, defaults to "log.json".
        """
        log_path = os.path.join(Meta.log_path, log_filename)
        with open(log_path, "w") as f:
            json.dump(self.run_log, f)
