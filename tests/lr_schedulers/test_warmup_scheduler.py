import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_step_scheduler(caplog):
    """Unit test of step scheduler"""

    caplog.set_level(logging.INFO)

    dirpath = "temp_test_scheduler"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test warmup steps
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": None,
                "warmup_steps": 2,
                "warmup_unit": "batch",
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 0

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    Meta.reset()
    emmental.init(dirpath)

    # Test warmup percentage
    config = {
        "learner_config": {
            "n_epochs": 4,
            "optimizer_config": {"optimizer": "sgd", "lr": 10},
            "lr_scheduler_config": {
                "lr_scheduler": None,
                "warmup_percentage": 0.5,
                "warmup_unit": "epoch",
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 0

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 5) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3)
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 10) < 1e-5

    shutil.rmtree(dirpath)
