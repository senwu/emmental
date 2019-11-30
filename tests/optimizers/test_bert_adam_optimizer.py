import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_bert_adam_optimizer(caplog):
    """Unit test of BertAdam optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "bert_adam"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default BertAdam setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.0,
    }

    # Test new BertAdam setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {"betas": (0.8, 0.9), "eps": 1e-05},
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "betas": (0.8, 0.9),
        "eps": 1e-05,
        "weight_decay": 0.05,
    }

    shutil.rmtree(dirpath)
