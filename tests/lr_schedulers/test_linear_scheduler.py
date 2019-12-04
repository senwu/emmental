import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_linear_scheduler(caplog):
    """Unit test of linear scheduler"""

    caplog.set_level(logging.INFO)

    lr_scheduler = "linear"
    dirpath = "temp_test_scheduler"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test per batch
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {"lr_scheduler": lr_scheduler},
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 10

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 7.5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 2.5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"]) < 1e-5

    # Test every 2 batches
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": lr_scheduler,
                "lr_scheduler_step_freq": 2,
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
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 7.5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 7.5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 5.0) < 1e-5

    shutil.rmtree(dirpath)
