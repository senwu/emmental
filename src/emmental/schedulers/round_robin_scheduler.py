"""Emmental round robin scheduler."""
import random
from typing import Iterator, List, Union

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.schedulers.scheduler import Batch, Scheduler


class RoundRobinScheduler(Scheduler):
    """Generate batch generator from all dataloaders in round robin order.

    Args:
      fillup: Whether fillup to make all dataloader the same size.
    """

    def __init__(self, fillup: bool = False) -> None:
        """Initialize RoundRobinScheduler."""
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
    ) -> Iterator[Union[Batch, List[Batch]]]:
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

        dataloader_indexer = []
        for idx, count in enumerate(batch_counts):
            dataloader_indexer.extend([idx] * count)

        random.shuffle(dataloader_indexer)

        for data_loader_idx in dataloader_indexer:
            uid_name = uid_names[data_loader_idx]
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

            yield Batch(
                X_dict[uid_name],
                X_dict,
                Y_dict,
                task_to_label_dicts[data_loader_idx],
                data_names[data_loader_idx],
                splits[data_loader_idx],
            )
