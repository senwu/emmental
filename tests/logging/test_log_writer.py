"""Emmental log writer unit tests."""
import json
import logging
import os

import yaml

import emmental
from emmental.logging.json_writer import JsonWriter
from emmental.logging.tensorboard_writer import TensorBoardWriter


def test_json_writer(caplog):
    """Unit test of log_writer."""
    caplog.set_level(logging.INFO)

    emmental.Meta.reset()

    emmental.init()
    emmental.Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_freq": 2},
            }
        }
    )

    json_writer = JsonWriter()

    json_writer.add_scalar(name="step 1", value=0.1, step=1)
    json_writer.add_scalar(name="step 2", value=0.2, step=2)

    config_filename = "config.yaml"
    json_writer.write_config(config_filename)

    # Test config
    with open(os.path.join(emmental.Meta.log_path, config_filename), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert config["meta_config"]["verbose"] is True
    assert config["logging_config"]["counter_unit"] == "sample"
    assert config["logging_config"]["checkpointing"] is True

    log_filename = "log.json"
    json_writer.write_log(log_filename)

    # Test log
    with open(os.path.join(emmental.Meta.log_path, log_filename), "r") as f:
        log = json.load(f)

    assert log == {"step 1": [[1, 0.1]], "step 2": [[2, 0.2]]}

    json_writer.close()


def test_tensorboard_writer(caplog):
    """Unit test of log_writer."""
    caplog.set_level(logging.INFO)

    emmental.Meta.reset()

    emmental.init()

    log_writer = TensorBoardWriter()

    log_writer.add_scalar(name="step 1", value=0.1, step=1)
    log_writer.add_scalar(name="step 2", value=0.2, step=2)

    config_filename = "config.yaml"
    log_writer.write_config(config_filename)

    # Test config
    with open(os.path.join(emmental.Meta.log_path, config_filename), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert config["meta_config"]["verbose"] is True
    assert config["logging_config"]["counter_unit"] == "epoch"
    assert config["logging_config"]["checkpointing"] is False

    log_writer.write_log()

    log_writer.close()
