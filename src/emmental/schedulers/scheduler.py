from abc import ABC, abstractmethod
from typing import Any, Iterator, List

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel


class Scheduler(ABC):
    r"""Generate batch generator from all dataloaders in designed order for MTL
      training.

    """

    def __init__(self) -> None:

        pass

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        r"""Get total number of batches per epoch.

        Args:
          dataloaders(list): List of dataloaders.

        Returns:
          int: Total number of batches per epoch.

        """

        raise NotImplementedError()

    @abstractmethod
    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model: EmmentalModel = None
    ) -> Iterator[Any]:
        r"""Generate batch generator from all dataloaders in designed order for
        one epoch.

        Args:
          dataloaders(list): List of dataloaders.
          model(EmmentalModel): The training model, defaults to None.

        Returns:
          genertor: A generator of all batches.

        """

        raise NotImplementedError()
