#! /usr/bin/env python
import logging

from emmental.utils.logging.counter import Counter


def test_counter_sample(caplog):
    """Unit test of counter (sample)"""

    caplog.set_level(logging.INFO)

    counter = Counter(
        {
            "logging_config": {
                "counter_unit": "sample",
                "evaluation_freq": 10,
                "checkpointing_freq": 2,
            }
        },
        n_batches_per_epoch=10,
    )

    counter.update(5)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    counter.update(5)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is False

    counter.update(10)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is True

    counter.update(5)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    assert counter.sample_count == 5
    assert counter.sample_total == 25

    assert counter.batch_total == 4
    assert counter.epoch_total == 0.4


def test_counter_batch(caplog):
    """Unit test of counter (batch)"""

    caplog.set_level(logging.INFO)

    counter = Counter(
        {
            "logging_config": {
                "counter_unit": "batch",
                "evaluation_freq": 2,
                "checkpointing_freq": 2,
            }
        },
        n_batches_per_epoch=5,
    )

    counter.update(5)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    counter.update(5)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is False

    counter.update(10)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    counter.update(5)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is True

    assert counter.batch_count == 0

    assert counter.sample_total == 25
    assert counter.batch_total == 4
    assert counter.epoch_total == 0.8


def test_counter_epoch(caplog):
    """Unit test of counter (epoch)"""

    caplog.set_level(logging.INFO)

    counter = Counter(
        {
            "logging_config": {
                "counter_unit": "epoch",
                "evaluation_freq": 1,
                "checkpointing_freq": 2,
            }
        },
        n_batches_per_epoch=2,
    )

    counter.update(5)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    counter.update(5)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is False

    counter.update(10)
    assert counter.trigger_evaluation() is False
    assert counter.trigger_checkpointing() is False

    counter.update(5)
    assert counter.trigger_evaluation() is True
    assert counter.trigger_checkpointing() is True

    assert counter.epoch_count == 0

    assert counter.sample_total == 25
    assert counter.batch_total == 4
    assert counter.epoch_total == 2
