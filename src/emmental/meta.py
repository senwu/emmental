"""Emmental meta."""
import logging
import math
import os
import tempfile
import uuid
from builtins import object
from datetime import datetime
from typing import Any, Dict, Optional, Type

import torch
import yaml

from emmental.utils.seed import set_random_seed
from emmental.utils.utils import merge

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
    local_rank: int = -1,
) -> None:
    """Initialize the logging and configuration.

    Args:
      log_dir: The directory to store logs in, defaults to tempfile.gettempdir().
      log_name: The log file name, defaults to "emmental.log".
      use_exact_log_path: Whether to use the exact log directory, defaults to False.
      format: The logging format string to use,
        defaults to "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s".
      level: The logging level to use, defaults to logging.INFO.
      config: The new configuration, defaults to {}.
      config_dir: The path to the config file, defaults to None.
      config_name: The config file name, defaults to "emmental-config.yaml".
      local_rank: local_rank for distributed training on gpus.
    """
    init_logging(log_dir, log_name, use_exact_log_path, format, level, local_rank)
    init_config()
    if config or config_dir is not None:
        Meta.update_config(config, config_dir, config_name, update_random_seed=False)

    set_random_seed(Meta.config["meta_config"]["seed"])
    Meta.check_config()


def init_config() -> None:
    """Load the default configuration."""
    # Load the default setting
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "emmental-default-config.yaml"
    )
    with open(default_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Loading Emmental default config from {default_config_path}.")

    Meta.config = config
    Meta.check_config()


def init_logging(
    log_dir: str = tempfile.gettempdir(),
    log_name: str = "emmental.log",
    use_exact_log_path: bool = False,
    format: str = "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level: int = logging.INFO,
    local_rank: int = -1,
) -> None:
    """Config logging to output to the provided log_dir.

    Will use a nested directory whose name is the current timestamp.

    Args:
      log_dir: The directory to store logs in, defaults to tempfile.gettempdir().
      log_name: The log file name, defaults to "emmental.log".
      use_exact_log_path: Whether to use the exact log directory, defaults to False.
      format: The logging format string to use,
        defaults to "[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s".
      level: The logging level to use, defaults to logging.INFO.
      local_rank: local_rank for distributed training on gpus.
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
        if local_rank in [-1, 0]:
            os.makedirs(log_path, exist_ok=True)

        # Configure the logger using the provided path
        if local_rank in [-1, 0]:
            logging.basicConfig(
                format=format,
                level=level,
                handlers=[
                    logging.FileHandler(os.path.join(log_path, log_name)),
                    logging.StreamHandler(),
                ],
            )
        else:
            logging.basicConfig(
                format=format, level=logging.WARN, handlers=[logging.StreamHandler()]
            )

        # Notify user of log location
        logger.info(f"Setting logging directory to: {log_path}")
        Meta.log_path = log_path
    else:
        logger.info(
            f"Logging was already initialized to use {Meta.log_path}.  "
            "To configure logging manually, call emmental.init_logging before "
            "initializing Meta."
        )


class Meta(object):
    """Singleton-like metadata class for all global variables.

    Adapted from the Unique Design Pattern:
        https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python
    """

    log_path: Optional[str] = None
    config: Optional[Dict[Any, Any]] = None

    @classmethod
    def init(cls) -> Type["Meta"]:
        """Initialize Meta."""
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
        update_random_seed: Optional[bool] = True,
    ) -> None:
        """Update the config with the configs in root of project and its parents.

        Note: There are two ways to update the config:
            (1) uses a config dict to update to config
            (2) uses path and filename to load yaml file to update config

        Args:
          config: The new configuration, defaults to {}.
          path: The path to the config file, defaults to os.getcwd().
          filename: The config file name, defaults to "emmental-config.yaml".
          update_random_seed: Whether update the random seed or not.
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
        if update_random_seed:
            set_random_seed(Meta.config["meta_config"]["seed"])

        Meta.check_config()

    @staticmethod
    def reset() -> None:
        """Clear shared variables of shared, global singleton."""
        Meta.log_path = None
        Meta.config = None

    @staticmethod
    def init_distributed_backend() -> None:
        """Initialize distributed learning backend."""
        if (
            Meta.config["learner_config"]["local_rank"] != -1
            and Meta.config["model_config"]["device"] != -1
        ):
            torch.cuda.set_device(Meta.config["learner_config"]["local_rank"])
            Meta.config["model_config"]["device"] = torch.device(
                "cuda", Meta.config["learner_config"]["local_rank"]
            )
            torch.distributed.init_process_group(
                backend=Meta.config["model_config"]["distributed_backend"]
            )

    @staticmethod
    def check_config() -> None:
        """Sanity check the config."""
        if Meta.config["logging_config"]["evaluation_freq"] == int(
            Meta.config["logging_config"]["evaluation_freq"]
        ):
            Meta.config["logging_config"]["evaluation_freq"] = int(
                Meta.config["logging_config"]["evaluation_freq"]
            )

        if (
            Meta.config["logging_config"]["counter_unit"]
            in [
                "sample",
                "batch",
            ]
            and isinstance(Meta.config["logging_config"]["evaluation_freq"], float)
        ):
            original_evaluation_freq = Meta.config["logging_config"]["evaluation_freq"]
            new_evaluation_freq = max(
                1, math.ceil(Meta.config["logging_config"]["evaluation_freq"])
            )
            logger.warning(
                f"Cannot use float value for evaluation_freq when "
                "counter_unit uses ['sample', 'batch'], switch "
                f"{original_evaluation_freq} to {new_evaluation_freq}."
            )

            Meta.config["logging_config"]["evaluation_freq"] = new_evaluation_freq

        if (
            Meta.config["logging_config"]["counter_unit"]
            in [
                "epoch",
            ]
            and isinstance(Meta.config["logging_config"]["evaluation_freq"], int)
            and Meta.config["logging_config"]["writer_config"]["write_loss_per_step"]
        ):
            logger.warning(
                "Cannot log loss per step when count_unit is epoch and "
                "evaluation_freq is int, switch write_loss_per_step to False."
            )

            Meta.config["logging_config"]["writer_config"][
                "write_loss_per_step"
            ] = False
