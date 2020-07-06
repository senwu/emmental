"""Emmental Adadelta optimizer unit tests."""
import logging
import shutil

from torch import nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from tests.utils import isequal

logger = logging.getLogger(__name__)


def test_adadelta_optimizer(caplog):
    """Unit test of Adadelta optimizer."""
    caplog.set_level(logging.INFO)

    optimizer = "adadelta"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default Adadelta setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {"lr": 0.001, "rho": 0.9, "eps": 1e-06, "weight_decay": 0},
    )

    # Test new Adadelta setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {"rho": 0.6, "eps": 1e-05},
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert isequal(
        emmental_learner.optimizer.defaults,
        {"lr": 0.02, "rho": 0.6, "eps": 1e-05, "weight_decay": 0.05},
    )

    shutil.rmtree(dirpath)
