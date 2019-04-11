import logging
import os
import tempfile
from builtins import object
from datetime import datetime

logger = logging.getLogger(__name__)


def init_logging(
    log_dir=tempfile.gettempdir(),
    log_name="emmental.log",
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
):
    """Configures logging to output to the provided log_dir.
    Will use a nested directory whose name is the current timestamp.
    :param log_dir: The directory to store logs in.
    :type log_dir: str
    :param format: The logging format string to use.
    :type format: str
    :param level: The logging level to use, e.g., logging.INFO.
    """

    if not Meta.log_path:
        # Generate a new directory using the log_dir, if it doesn't exist
        date = datetime.now().strftime("%Y_%m_%d")
        time = datetime.now().strftime("%H_%M_%S")
        log_path = os.path.join(log_dir, date, time)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

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
    """Singleton-like metadata class for all global variables.
    Adapted from the Unique Design Pattern:
    https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python
    """

    log_path = None

    @classmethod
    def init(cls):
        """Return the unique Meta class."""
        if not Meta.log_path:
            init_logging()

        return cls
