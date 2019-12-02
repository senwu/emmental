import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_exponential_scheduler(caplog):
    """Unit test of exponential scheduler"""

    caplog.set_level(logging.INFO)

    lr_scheduler = "exponential"
    dirpath = "temp_test_scheduler"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test step per batch
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": lr_scheduler,
                "exponential_config": {"gamma": 0.1},
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 10

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.01) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.001) < 1e-5

    # Test step per epoch
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": lr_scheduler,
                "lr_scheduler_step_unit": "epoch",
                "exponential_config": {"gamma": 0.1},
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 2
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 10

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.1) < 1e-5

    shutil.rmtree(dirpath)
