import logging
import os

import yaml

MAX_CONFIG_SEARCH_DEPTH = 25  # Max num of parent directories to look for config
logger = logging.getLogger(__name__)


def _merge(x, y):
    """Merge two nested dictionaries. Overwrite values in x with values in y."""

    merged = {**x, **y}

    xkeys = x.keys()

    for key in xkeys:
        if isinstance(x[key], dict) and key in y:
            merged[key] = _merge(x[key], y[key])

    return merged


def load_config(path=os.getcwd(), filename="emmental-config.yaml"):
    """Load configs in root of project and its parents.

    :param path: the path to the config file, defaults to os.getcwd()
    :param path: str, optional
    :param filename: the config file name, defaults to "emmental-config.yaml"
    :param filename: str, optional
    :return: the config object
    :rtype: dict
    """

    # Load the default setting
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "emmental-default-config.yaml"
    )
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loading Emmental default config from {default_config_path}.")

    # Update the default setting with given config
    tries = 0
    current_dir = path
    while current_dir and tries < MAX_CONFIG_SEARCH_DEPTH:
        potential_path = os.path.join(current_dir, filename)
        print(potential_path, os.path.exists(potential_path))
        if os.path.exists(potential_path):
            with open(potential_path, "r") as f:
                config = _merge(config, yaml.safe_load(f))
            logger.debug(f"Loading Emmental config from {potential_path}.")
            break

        new_dir = os.path.split(current_dir)[0]
        if current_dir == new_dir:
            logger.debug("Unable to find config file. Using defaults.")
            break
        current_dir = new_dir
        tries += 1

    return config
