"""Emmental plateau scheduler unit tests."""
import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_plateau_scheduler(caplog):
    """Unit test of plateau scheduler."""
    caplog.set_level(logging.INFO)

    lr_scheduler = "plateau"
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
                "plateau_config": {
                    "metric": "model/train/all/loss",
                    "mode": "min",
                    "factor": 0.1,
                    "patience": 1,
                    "threshold": 0.0001,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "eps": 1e-08,
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
    emmental_learner._update_lr_scheduler(model, 0, {"model/train/all/loss": 1})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {"model/train/all/loss": 1})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {"model/train/all/loss": 1})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {"model/train/all/loss": 0.1})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    shutil.rmtree(dirpath)
