import logging
import shutil

import torch.nn as nn

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner

logger = logging.getLogger(__name__)


def test_r_prop_optimizer(caplog):
    """Unit test of Rprop optimizer"""

    caplog.set_level(logging.INFO)

    optimizer = "r_prop"
    dirpath = "temp_test_optimizer"
    model = nn.Linear(1, 1)
    emmental_learner = EmmentalLearner()

    Meta.reset()
    emmental.init(dirpath)

    # Test default Rprop setting
    config = {"learner_config": {"optimizer_config": {"optimizer": optimizer}}}
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.001,
        "etas": (0.5, 1.2),
        "step_sizes": (1e-06, 50),
    }

    # Test new Rprop setting
    config = {
        "learner_config": {
            "optimizer_config": {
                "optimizer": optimizer,
                "lr": 0.02,
                "l2": 0.05,
                f"{optimizer}_config": {"etas": (0.3, 1.5), "step_sizes": (1e-04, 30)},
            }
        }
    }
    emmental.Meta.update_config(config)
    emmental_learner._set_optimizer(model)

    assert emmental_learner.optimizer.defaults == {
        "lr": 0.02,
        "etas": (0.3, 1.5),
        "step_sizes": (1e-04, 30),
    }

    shutil.rmtree(dirpath)
