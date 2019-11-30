import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_rms_prop_optimizer(caplog):
    """Unit test of RMSprop optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "rms_prop"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default RMSprop setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "alpha": 0.99,
        "eps": 1e-08,
        "momentum": 0,
        "centered": False,
        "weight_decay": 0,
    }

    # Test new RMSprop setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "alpha": 0.9,
                    "eps": 1e-05,
                    "momentum": 0.1,
                    "centered": True,
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "alpha": 0.9,
        "eps": 1e-05,
        "momentum": 0.1,
        "centered": True,
        "weight_decay": 0.05,
    }

    shutil.rmtree(dirpath)
