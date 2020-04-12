import logging
import os
import shutil

import pytest

import emmental
from emmental.logging.checkpointer import Checkpointer


def test_checkpointer_specific_path(caplog):
    """Unit test of checkpointer"""

    caplog.set_level(logging.INFO)

    checkpoint_path = "temp_test_checkpointer"

    emmental.Meta.reset()

    emmental.init()
    emmental.Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing": True,
                "checkpointer_config": {
                    "checkpoint_freq": 2,
                    "checkpoint_path": checkpoint_path,
                },
            }
        }
    )

    checkpointer = Checkpointer()
    checkpointer.clear()

    assert os.path.exists(checkpoint_path) is True
    shutil.rmtree(checkpoint_path)


def test_checkpointer_wrong_freq(caplog):
    """Unit test of checkpointer (wrong frequency)"""

    caplog.set_level(logging.INFO)

    emmental.Meta.reset()

    emmental.init()
    emmental.Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_freq": -1},
            }
        }
    )

    with pytest.raises(ValueError):
        checkpointer = Checkpointer()
        checkpointer.close()


def test_checkpointer_wrong_metric_mode(caplog):
    """Unit test of checkpointer (wrong metric mode)"""

    caplog.set_level(logging.INFO)

    emmental.Meta.reset()

    emmental.init()
    emmental.Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing": True,
                "checkpointer_config": {
                    "checkpoint_metric": {"model/all/train/loss": "min1"}
                },
            }
        }
    )

    with pytest.raises(ValueError):
        checkpointer = Checkpointer()
        checkpointer.close()
