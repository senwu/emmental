"""Emmental sequential scheduler."""
from typing import Dict, Iterator, List, Tuple, Union

from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.schedulers.scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Generate batch generator from all dataloaders in sequential order.

    Args:
      fillup: Whether fillup to make all dataloader the same size.
    """

    def __init__(self, fillup: bool = False) -> None:
        """Initialize SequentialScheduler."""
        super().__init__()

        self.fillup = fillup

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        """Get total number of batches per epoch.

        Args:
          dataloaders: List of dataloaders.

        Returns:
          Total number of batches per epoch.
        """
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        return sum(batch_counts)

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model: EmmentalModel = None
    ) -> Iterator[
        Tuple[
            List[str],
            Dict[str, Union[Tensor, List[str]]],
            Dict[str, Tensor],
            Dict[str, str],
            str,
            str,
        ]
    ]:
        """Generate batch generator from all dataloaders for one epoch.

        Args:
          dataloaders: List of dataloaders.
          model: The training model, defaults to None.

        Returns:
          A generator of all batches.
        """
        task_to_label_dicts = [
            dataloader.task_to_label_dict for dataloader in dataloaders
        ]
        uid_names = [dataloader.uid for dataloader in dataloaders]
        data_names = [dataloader.data_name for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        # Calc the batch size for each dataloader
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        if self.fillup:
            batch_counts = [max(batch_counts)] * len(dataloaders)

        for idx in range(len(dataloaders)):
            if dataloaders[idx].n_batches:
                batch_counts[idx] = dataloaders[idx].n_batches

        for (
            data_loader_idx,
            (task_to_label_dict, data_name, batch_count, split, uid_name),
        ) in enumerate(
            zip(task_to_label_dicts, data_names, batch_counts, splits, uid_names)
        ):
            for batch_idx in range(batch_count):
                try:
                    batch = next(data_loaders[data_loader_idx])
                except StopIteration:
                    data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                    batch = next(data_loaders[data_loader_idx])

                if not isinstance(batch, dict):
                    X_dict, Y_dict = batch
                else:
                    X_dict = batch
                    Y_dict = None

                yield X_dict[
                    uid_name
                ], X_dict, Y_dict, task_to_label_dict, data_name, split
