"""Emmental meta unit tests."""
import logging
import os
import shutil

import emmental
from emmental import Meta


def test_meta(caplog):
    """Unit test of meta."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_meta_log_folder"

    Meta.reset()
    emmental.init(dirpath)

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

    emmental.Meta.update_config(
        path="tests/shared", filename="emmental-test-config.yaml"
    )
    assert Meta.config["meta_config"] == {
        "seed": 1,
        "verbose": False,
        "log_path": "tests",
        "use_exact_log_path": False,
    }

    # Test unable to find config file
    Meta.reset()
    emmental.init(dirpath)

    emmental.Meta.update_config(path=os.path.dirname(__file__))
    assert Meta.config["meta_config"] == {
        "seed": None,
        "verbose": True,
        "log_path": "logs",
        "use_exact_log_path": False,
    }

    # Remove the temp folder
    shutil.rmtree(dirpath)
