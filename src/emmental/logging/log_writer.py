import json
import os
from collections import defaultdict

import yaml

from emmental.meta import Meta


class LogWriter(object):
    """A class for logging during training process.
    """

    def __init__(self):
        """Initialize the log writer.
        """

        self.run_log = defaultdict(list)

    def add_config(self, config):
        """Log config.

        :param config: The config
        :type config: dict
        """

        self.config = config

    def add_scalar(self, name, value, step):
        """Log a scalar variable.

        :param name: The name of the scalar
        :type name: str
        :param value: The value of the scalar
        :type value: float or int
        :param step: The current step
        :type step: float or int
        """

        self.run_log[name].append((step, value))

    def write_config(self, config_filename="config.yaml"):
        """Dump the config to file

        :param config_filename: The config filename, defaults to "config.yaml"
        :type config_filename: str, optional
        """

        config_path = os.path.join(Meta.log_path, config_filename)
        with open(config_path, "w") as yml:
            yaml.dump(self.config, yml, default_flow_style=False, allow_unicode=True)

    def write_log(self, log_filename="log.json"):
        """Dump the log to file.

        :param log_filename: The log filename, defaults to "log.json"
        :type log_filename: str, optional
        """

        log_path = os.path.join(Meta.log_path, log_filename)
        print(log_path)
        with open(log_path, "w") as f:
            json.dump(self.run_log, f)

    def close(self):
        """Close the log writer.
        """
        pass
