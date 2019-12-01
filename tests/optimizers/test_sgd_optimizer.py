import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_sgd_optimizer(caplog):
    """Unit test of SGD optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "sgd"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default SGD setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "momentum": 0,
        "dampening": 0,
        "nesterov": False,
        "weight_decay": 0.0,
    }

    # Test new SGD setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "momentum": 0.1,
                    "dampening": 0,
                    "nesterov": True,
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "momentum": 0.1,
        "dampening": 0,
        "nesterov": True,
        "weight_decay": 0.05,
    }

    shutil.rmtree(dirpath)
