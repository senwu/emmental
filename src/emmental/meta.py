import logging
import os
import tempfile
import uuid
from builtins import object
from datetime import datetime
from typing import Any, Dict, Optional, Type

import yaml

from emmental.utils.utils import merge, set_random_seed

MAX_CONFIG_SEARCH_DEPTH = 25  # Max num of parent directories to look for config

logger = logging.getLogger(__name__)


def init(
    log_dir: str = tempfile.gettempdir(),
    log_name: str = "emmental.log",
    use_exact_log_path: bool = False,
    format: str = "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level: int = logging.INFO,
    config: Optional[Dict[Any, Any]] = {},
    config_dir: Optional[str] = None,
    config_name: Optional[str] = "emmental-config.yaml",
) -> None:
    r"""Initialize the logging and configuration.

    Args:
      log_dir(str, optional): The directory to store logs in,
        defaults to tempfile.gettempdir().
      log_name(str, optional): The log file name, defaults to "emmental.log".
      use_exact_log_path(bool, optional): Whether to use the exact log directory,
        defaults to False.
      format(str, optional): The logging format string to use,
        defaults to "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s".
      level(int, optional): The logging level to use, defaults to logging.INFO.
      config(dict, optional): The new configuration, defaults to {}.
      config_dir(str, optional): The path to the config file, defaults to None.
      config_name(str, optional): The config file name,
        defaults to "emmental-config.yaml".

    """

    init_logging(log_dir, log_name, use_exact_log_path, format, level)
    init_config()
    if config or config_dir is not None:
        Meta.update_config(config, config_dir, config_name)

    set_random_seed(Meta.config["meta_config"]["seed"])


def init_config() -> None:
    r"""Load the default configuration."""

    # Load the default setting
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "emmental-default-config.yaml"
    )
    with open(default_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Loading Emmental default config from {default_config_path}.")

    Meta.config = config


def init_logging(
    log_dir: str = tempfile.gettempdir(),
    log_name: str = "emmental.log",
    use_exact_log_path: bool = False,
    format: str = "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level: int = logging.INFO,
) -> None:
    r"""Configures logging to output to the provided log_dir.
      Will use a nested directory whose name is the current timestamp.

    Args:
      log_dir(str, optional): The directory to store logs in,
        defaults to tempfile.gettempdir().
      log_name(str, optional): The log file name, defaults to "emmental.log".
      use_exact_log_path(bool, optional): Whether to use the exact log directory,
        defaults to False.
      format(str, optional): The logging format string to use,
        defaults to "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s".
      level(int, optional): The logging level to use, defaults to logging.INFO.

    """

    if not Meta.log_path:
        if not use_exact_log_path:
            # Generate a new directory using the log_dir, if it doesn't exist
            date = datetime.now().strftime("%Y_%m_%d")
            time = datetime.now().strftime("%H_%M_%S")
            uid = str(uuid.uuid4())[:8]
            log_path = os.path.join(log_dir, date, time, uid)
            while os.path.exists(log_path):
                uid = str(uuid.uuid4())[:8]
                log_path = os.path.join(log_dir, date, time, uid)
        else:
            log_path = log_dir
        os.makedirs(log_path, exist_ok=True)

        # Configure the logger using the provided path
        logging.basicConfig(
            format=format,
            level=level,
            handlers=[
                logging.FileHandler(os.path.join(log_path, log_name)),
                logging.StreamHandler(),
            ],
        )

        # Notify user of log location
        logger.info(f"Setting logging directory to: {log_path}")
        Meta.log_path = log_path
    else:
        logger.info(
            f"Logging was already initialized to use {Meta.log_path}.  "
            "To configure logging manually, call emmental.init_logging before "
            "initialiting Meta."
        )


class Meta(object):
    r"""Singleton-like metadata class for all global variables.
      Adapted from the Unique Design Pattern:
        https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python

    """

    log_path: Optional[str] = None
    config: Optional[Dict[Any, Any]] = None

    @classmethod
    def init(cls) -> Type["Meta"]:
        """ """
        if not Meta.log_path:
            init_logging()

        if not Meta.config:
            init_config()

        return cls

    @staticmethod
    def update_config(
        config: Optional[Dict[Any, Any]] = {},
        path: Optional[str] = None,
        filename: Optional[str] = "emmental-config.yaml",
    ) -> None:
        r"""Update the configuration with the configs in root of project and
          its parents.

        Note: There are two ways to update the config:
            (1) uses a config dict to update to config
            (2) uses path and filename to load yaml file to update config

        Args:
          config(dict, optional): The new configuration, defaults to {}.
          path(str, optional): The path to the config file, defaults to os.getcwd().
          filename(str, optional): The config file name,
            defaults to "emmental-config.yaml".

        """

        if config != {}:
            Meta.config = merge(Meta.config, config, specical_keys="checkpoint_metric")
            logger.info("Updating Emmental config from user provided config.")

        if path is not None:
            tries = 0
            current_dir = path
            while current_dir and tries < MAX_CONFIG_SEARCH_DEPTH:
                potential_path = os.path.join(current_dir, filename)
                if os.path.exists(potential_path):
                    with open(potential_path, "r") as f:
                        Meta.config = merge(
                            Meta.config,
                            yaml.load(f, Loader=yaml.FullLoader),
                            specical_keys="checkpoint_metric",
                        )
                    logger.info(f"Updating Emmental config from {potential_path}.")
                    break

                new_dir = os.path.split(current_dir)[0]
                if current_dir == new_dir:
                    logger.info("Unable to find config file. Using defaults.")
                    break
                current_dir = new_dir
                tries += 1

    @staticmethod
    def reset() -> None:
        r"""Clears shared variables of shared, global singleton."""

        Meta.log_path = None
        Meta.config = None
