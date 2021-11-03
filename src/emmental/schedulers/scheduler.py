"""Emmental scheduler."""
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union

from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel


class Batch:
    """A batch of examples along with its meta information.

    Args:
      uids: The uids of samples.
      X_dict: The input feature/data of samples.
      Y_dict: The output/label of samples.
      task_to_label_dict: The task to label mapping.
      data_name: The name of the dataset that samples come from.
      split: The split information, defaults to "train".
    """

    def __init__(
        self,
        uids: List[str],
        X_dict: Dict[str, Union[Tensor, List[str]]],
        Y_dict: Dict[str, Tensor],
        task_to_label_dict: Dict[str, str],
        data_name: str,
        split: str = "train",
    ) -> None:
        """Initialize Batch."""
        self.uids = uids
        self.X_dict = X_dict
        self.Y_dict = Y_dict
        self.task_to_label_dict = task_to_label_dict
        self.data_name = data_name
        self.split = split

    def __repr__(self) -> str:
        """Represent the batch as a string."""
        return (
            f"Batch(uids={self.uids}, "
            f"X_dict={self.X_dict}, "
            f"Y_dict={self.Y_dict}, "
            f"task_to_label_dict={self.task_to_label_dict}, "
            f"data_name={self.data_name}, "
            f"split={self.split})"
        )


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
    ) -> Iterator[Union[Batch, List[Batch]]]:
        """Generate batch generator from all dataloaders for one epoch.

        Args:
          dataloaders: List of dataloaders.
          model: The training model, defaults to None.

        Returns:
          A generator of all batches.
        """
        raise NotImplementedError()
