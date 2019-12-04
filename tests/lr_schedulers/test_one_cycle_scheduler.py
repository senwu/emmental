import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_one_cycle_scheduler(caplog):
    """Unit test of one cycle scheduler"""

    caplog.set_level(logging.INFO)

    lr_scheduler = "one_cycle"
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
                "one_cycle_config": {
                    "anneal_strategy": "cos",
                    "base_momentum": 0.85,
                    "cycle_momentum": True,
                    "div_factor": 1,
                    "final_div_factor": 10000.0,
                    "last_epoch": -1,
                    "max_lr": 0.1,
                    "max_momentum": 0.95,
                    "pct_start": 0.3,
                },
            },
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner.n_batches_per_epoch = 1
    emmental_learner._set_optimizer(model)
    emmental_learner._set_lr_scheduler(model)

    assert emmental_learner.optimizer.param_groups[0]["lr"] == 0.1

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 0, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.08117637264392738)
        < 1e-5
    )

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 1, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.028312982462817687)
        < 1e-5
    )

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 2, {})
    assert abs(emmental_learner.optimizer.param_groups[0]["lr"] - 1e-05) < 1e-5

    emmental_learner.optimizer.step()
    emmental_learner._update_lr_scheduler(model, 3, {})
    assert (
        abs(emmental_learner.optimizer.param_groups[0]["lr"] - 0.028312982462817677)
        < 1e-5
    )

    shutil.rmtree(dirpath)
