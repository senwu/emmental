"""Emmental cyclic scheduler unit tests."""
import logging
import shutil

from torch import nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_cyclic_scheduler(caplog):
    """Unit test of cyclic scheduler."""
    caplog.set_level(logging.INFO)

    lr_scheduler = "cyclic"
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
                "cyclic_config": {
                    "base_lr": 10,
                    "base_momentum": 0.8,
                    "cycle_momentum": True,
                    "gamma": 1.0,
                    "last_epoch": -1,
                    "max_lr": 0.1,
                    "max_momentum": 0.9,
                    "mode": "triangular",
                    "scale_fn": None,
                    "scale_mode": "cycle",
                    "step_size_down": None,
                    "step_size_up": 2000,
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
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 9.995049999999999) < 1e-5
    )

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 9.990100000000002) < 1e-5
    )

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 9.985149999999999) < 1e-5
    )

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 9.980200000000002) < 1e-5
    )

    shutil.rmtree(dirpath)
