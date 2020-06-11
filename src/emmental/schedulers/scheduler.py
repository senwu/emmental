"""Emmental scheduler."""
from abc import ABC, abstractmethod
from typing import Any, Iterator, List

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel


class Scheduler(ABC):
    """Generate batch generator from dataloaders in designed order."""

    def __init__(self) -> None:
        """Initialize Scheduler."""
        pass

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        """Get total number of batches per epoch.

        Args:
          dataloaders: List of dataloaders.

        Returns:
          Total number of batches per epoch.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model: EmmentalModel = None
    ) -> Iterator[Any]:
        """Generate batch generator from all dataloaders for one epoch.

        Args:
          dataloaders: List of dataloaders.
          model: The training model, defaults to None.

        Returns:
          A generator of all batches.
        """
        raise NotImplementedError()
