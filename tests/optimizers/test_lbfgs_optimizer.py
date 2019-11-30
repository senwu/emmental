import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_lbfgs_optimizer(caplog):
    """Unit test of LBFGS optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "lbfgs"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default LBFGS setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "max_iter": 20,
        "max_eval": 25,
        "tolerance_grad": 1e-07,
        "tolerance_change": 1e-09,
        "history_size": 100,
        "line_search_fn": None,
    }

    # Test new LBFGS setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {
                    "max_iter": 30,
                    "max_eval": 40,
                    "tolerance_grad": 1e-04,
                    "tolerance_change": 1e-05,
                    "history_size": 10,
                    "line_search_fn": "strong_wolfe",
                },
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "max_iter": 30,
        "max_eval": 40,
        "tolerance_grad": 1e-04,
        "tolerance_change": 1e-05,
        "history_size": 10,
        "line_search_fn": "strong_wolfe",
    }

    shutil.rmtree(dirpath)
