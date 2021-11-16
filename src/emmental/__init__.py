"""Emmental package."""
from emmental._version import __version__
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.learner import EmmentalLearner
from emmental.meta import Meta, init
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import Action, EmmentalTask

__all__ = [
    "__version__",
    "Action",
    "EmmentalDataLoader",
    "EmmentalDataset",
    "EmmentalLearner",
    "EmmentalModel",
    "EmmentalTask",
    "EmmentalTaskFlowAction",
    "Meta",
    "Scorer",
    "init",
]
