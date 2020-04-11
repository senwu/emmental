import logging

import pytest

import emmental
from emmental.logging.logging_manager import LoggingManager
from emmental.meta import Meta


def test_logging_manager_sample(caplog):
    """Unit test of logging_manager (sample)"""

    caplog.set_level(logging.INFO)

    Meta.reset()

    emmental.init()
    Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_freq": 2},
            }
        }
    )

    logging_manager = LoggingManager(n_batches_per_epoch=10)

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(10)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is True

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    assert logging_manager.sample_count == 5
    assert logging_manager.sample_total == 25

    assert logging_manager.batch_total == 4
    assert logging_manager.epoch_total == 0.4


def test_logging_manager_batch(caplog):
    """Unit test of logging_manager (batch)"""

    caplog.set_level(logging.INFO)

    emmental.init()
    Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "batch",
                "evaluation_freq": 2,
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_freq": 2},
            }
        }
    )

    logging_manager = LoggingManager(n_batches_per_epoch=5)

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(10)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is True

    assert logging_manager.batch_count == 0

    assert logging_manager.sample_total == 25
    assert logging_manager.batch_total == 4
    assert logging_manager.epoch_total == 0.8


def test_logging_manager_epoch(caplog):
    """Unit test of logging_manager (epoch)"""

    caplog.set_level(logging.INFO)

    emmental.init()
    Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "epoch",
                "evaluation_freq": 1,
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_freq": 2},
            }
        }
    )

    logging_manager = LoggingManager(n_batches_per_epoch=2)

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(10)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is True

    assert logging_manager.epoch_count == 0

    assert logging_manager.sample_total == 25
    assert logging_manager.batch_total == 4
    assert logging_manager.epoch_total == 2


def test_logging_manager_no_checkpointing(caplog):
    """Unit test of logging_manager (no checkpointing)"""

    caplog.set_level(logging.INFO)

    emmental.init()
    Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "epoch",
                "evaluation_freq": 1,
                "checkpointing": False,
                "checkpointer_config": {"checkpoint_freq": 2},
                "writer_config": {"writer": None},
            }
        }
    )

    logging_manager = LoggingManager(n_batches_per_epoch=2)

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(10)
    assert logging_manager.trigger_evaluation() is False
    assert logging_manager.trigger_checkpointing() is False

    logging_manager.update(5)
    assert logging_manager.trigger_evaluation() is True
    assert logging_manager.trigger_checkpointing() is False

    assert logging_manager.epoch_count == 0

    assert logging_manager.sample_total == 25
    assert logging_manager.batch_total == 4
    assert logging_manager.epoch_total == 2


def test_logging_manager_wrong_counter_unit(caplog):
    """Unit test of logging_manager (no checkpointing)"""

    caplog.set_level(logging.INFO)

    emmental.init()
    Meta.update_config(
        config={
            "logging_config": {
                "counter_unit": "epochs",
                "evaluation_freq": 1,
                "checkpointing": False,
                "checkpointer_config": {"checkpoint_freq": 2},
            }
        }
    )

    with pytest.raises(ValueError):
        logging_manager = LoggingManager(n_batches_per_epoch=2)
        logging_manager.update(5)
