"""Emmental customzied optimizer unit tests."""
import logging
import shutil
from functools import partial

from torch import nn as nn, optim as optim

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from tests.utils import isequal

logger = logging.getLogger(__name__)


def test_customzied_optimizer(caplog):
    """Unit test of customzied optimizer."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    optimizer = optim.ASGD

    # Test default setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {
            "lr": 0.01,
            "lambd": 0.0001,
            "alpha": 0.75,
            "t0": 1_000_000.0,
            "weight_decay": 0,
        },
    )

    optimizer = partial(optim.ASGD, lr=0.001)

    # Test default setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {
            "lr": 0.001,
            "lambd": 0.0001,
            "alpha": 0.75,
            "t0": 1_000_000.0,
            "weight_decay": 0,
        },
    )

    shutil.rmtree(dirpath)
