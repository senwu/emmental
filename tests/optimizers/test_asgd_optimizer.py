"""Emmental ASGD optimizer unit tests."""
import logging
import shutil

from torch import nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from tests.utils import isequal

logger = logging.getLogger(__name__)


def test_asgd_optimizer(caplog):
    """Unit test of ASGD optimizer."""
    caplog.set_level(logging.INFO)

    optimizer = "asgd"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default ASGD setting
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

    # Test default ASGD setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {"lambd": 0.1, "alpha": 0.5, "t0": 1e5},
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {"lr": 0.02, "lambd": 0.1, "alpha": 0.5, "t0": 1e5, "weight_decay": 0.05},
    )

    shutil.rmtree(dirpath)
