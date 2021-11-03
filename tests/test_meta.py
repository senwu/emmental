"""Emmental meta unit tests."""
import logging
import os
import shutil

from emmental import Meta, init


def test_meta(caplog):
    """Unit test of meta."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_meta_log_folder"

    Meta.reset()
    init(dirpath)

    # Check the log folder is created correctly
    assert os.path.isdir(dirpath) is True
    assert Meta.log_path.startswith(dirpath) is True

    # Check the config is created
    assert isinstance(Meta.config, dict) is True
    assert Meta.config["meta_config"] == {
        "seed": None,
        "verbose": True,
        "log_path": "logs",
        "use_exact_log_path": False,
    }

    Meta.update_config(path="tests/shared", filename="emmental-test-config.yaml")
    assert Meta.config["meta_config"] == {
        "seed": 1,
        "verbose": False,
        "log_path": "tests",
        "use_exact_log_path": False,
    }

    # Test unable to find config file
    Meta.reset()
    init(dirpath)

    Meta.update_config(path=os.path.dirname(__file__))
    assert Meta.config["meta_config"] == {
        "seed": None,
        "verbose": True,
        "log_path": "logs",
        "use_exact_log_path": False,
    }

    # Remove the temp folder
    shutil.rmtree(dirpath)


def test_config_check_in_meta(caplog):
    """Unit test of config check in meta."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_meta_log_folder"

    Meta.reset()
    init(dirpath)

    config = {"logging_config": {"evaluation_freq": 5.0}}
    Meta.update_config(config)

    assert type(Meta.config["logging_config"]["evaluation_freq"]) == int
    assert Meta.config["logging_config"]["evaluation_freq"] == 5

    config = {"logging_config": {"counter_unit": "batch", "evaluation_freq": 2.3}}
    Meta.update_config(config)

    assert type(Meta.config["logging_config"]["evaluation_freq"]) == int
    assert Meta.config["logging_config"]["evaluation_freq"] == 3

    config = {"logging_config": {"counter_unit": "sample", "evaluation_freq": 0.2}}
    Meta.update_config(config)

    assert type(Meta.config["logging_config"]["evaluation_freq"]) == int
    assert Meta.config["logging_config"]["evaluation_freq"] == 1

    config = {
        "logging_config": {
            "counter_unit": "epoch",
            "evaluation_freq": 1,
            "writer_config": {"write_loss_per_step": True},
        }
    }
    Meta.update_config(config)

    assert (
        Meta.config["logging_config"]["writer_config"]["write_loss_per_step"] is False
    )

    # Remove the temp folder
    shutil.rmtree(dirpath)
