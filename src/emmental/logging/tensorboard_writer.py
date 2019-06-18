import json

from tensorboardX import SummaryWriter

from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(self):
        super().__init__()

        # Set up tensorboard summary writer
        self.writer = SummaryWriter(Meta.log_path)

    def add_scalar(self, name, value, step):
        """Log a scalar variable.

        :param name: The name of the scalar
        :type name: str
        :param value: The value of the scalar
        :type value: float or int
        :param step: The current step
        :type step: float or int
        """

        self.writer.add_scalar(name, value, step)

    def write_config(self, config_filename="config.yaml"):
        """Dump the config to file.

        :param config_filename: The config filename, defaults to "config.yaml"
        :type config_filename: str, optional
        """

        config = json.dumps(Meta.config)
        self.writer.add_text(tag="config", text_string=config)

        super().write_config(config_filename)

    def close(self):
        """Close the tensorboard writer.
        """

        self.writer.close()
