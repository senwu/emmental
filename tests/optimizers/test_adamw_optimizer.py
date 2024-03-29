"""Emmental AdamW optimizer unit tests."""
import logging
import shutil

from torch import nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_adamw_optimizer(caplog):
    """Unit test of AdamW optimizer."""
    caplog.set_level(logging.INFO)

    optimizer = "adamw"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default AdamW setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    optimizer_config = emmental_learner.optimizer.defaults

    if "maximize" in optimizer_config:
        del optimizer_config["maximize"]

    assert optimizer_config == {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "amsgrad": False,
        "weight_decay": 0,
    }

    # Test new AdamW setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "betas": (0.9, 0.99),
                    "eps": 1e-05,
                    "amsgrad": True,
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    optimizer_config = emmental_learner.optimizer.defaults

    if "maximize" in optimizer_config:
        del optimizer_config["maximize"]

    assert optimizer_config == {
        "lr": 0.02,
        "betas": (0.9, 0.99),
        "eps": 1e-05,
        "amsgrad": True,
        "weight_decay": 0.05,
    }

    shutil.rmtree(dirpath)
