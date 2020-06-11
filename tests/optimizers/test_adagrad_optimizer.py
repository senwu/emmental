"""Emmental Adagrad optimizer unit tests."""
import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from tests.utils import isequal

logger = logging.getLogger(__name__)


def test_adagrad_optimizer(caplog):
    """Unit test of Adagrad optimizer."""
    caplog.set_level(logging.INFO)

    optimizer = "adagrad"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default Adagrad setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {
            "lr": 0.001,
            "lr_decay": 0,
            "initial_accumulator_value": 0,
            "eps": 1e-10,
            "weight_decay": 0,
        },
    )

    # Test new Adagrad setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "lr_decay": 0.1,
                    "initial_accumulator_value": 0.2,
                    "eps": 1e-5,
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {
            "lr": 0.02,
            "lr_decay": 0.1,
            "initial_accumulator_value": 0.2,
            "eps": 1e-5,
            "weight_decay": 0.05,
        },
    )

    shutil.rmtree(dirpath)
