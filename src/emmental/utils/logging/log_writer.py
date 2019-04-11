import json
import os
from collections import defaultdict

import yaml

from emmental.meta import Meta


class LogWriter(object):
    """A class for logging during training process.

    :param object: [description]
    :type object: [type]
    """

    def __init__(self, config=None, verbose=True):
        self.verbose = verbose

        self.config = config
        self.run_log = defaultdict(list)

    def add_config(self, config):
        """Log config"""
        self.config = config

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""
        self.run_log[name].append((step, value))

    def write_config(self, config_filename="config.yaml"):
        """Dump the config to file"""

        config_path = os.path.join(Meta.log_path, config_filename)
        with open(config_path, "w") as yml:
            yaml.dump(self.config, yml, default_flow_style=False, allow_unicode=True)

    def write_log(self, log_filename="log.json"):
        """Dump the log to file"""

        log_path = os.path.join(Meta.log_path, log_filename)
        print(log_path)
        with open(log_path, "w") as f:
            json.dump(self.run_log, f)

    def close(self):
        pass
