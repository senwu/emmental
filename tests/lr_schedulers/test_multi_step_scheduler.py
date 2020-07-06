"""Emmental multi step scheduler unit tests."""
import logging
import shutil

from torch import nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_multi_step_scheduler(caplog):
    """Unit test of multi step scheduler."""
    caplog.set_level(logging.INFO)

    lr_scheduler = "multi_step"
    dirpath = "temp_test_scheduler"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": lr_scheduler,
                "multi_step_config": {
                    "milestones": [1, 3],
                    "gamma": 0.1,
                    "last_epoch": -1,
                },
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 10

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.1) < 1e-5

    shutil.rmtree(dirpath)
