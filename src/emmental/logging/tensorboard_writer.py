import json

from torch.utils.tensorboard import SummaryWriter

from emmental.logging.log_writer import LogWriter
from emmental.meta import Meta


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(self):
        super().__init__()

        # Set up tensorboard summary writer
        self.writer = SummaryWriter(Meta.log_path)

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""

        self.writer.add_scalar(name, value, step)

    def write_config(self, config_filename="config.yaml"):
        """Dump the config to file"""
        config = json.dumps(Meta.config)
        self.writer.add_text(tag="config", text_string=config)

        super().write_config(config_filename)

    def close(self):
        self.writer.close()
