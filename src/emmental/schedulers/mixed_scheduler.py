"""Emmental mixed scheduler."""
from typing import Iterator, List, Union

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.schedulers.scheduler import Batch, Scheduler


class MixedScheduler(Scheduler):
    """Generate batch generator from all dataloaders in mixture for MTL training.

    Args:
      fillup: Whether fillup to make all dataloader the same size.
    """

    def __init__(self, fillup: bool = False) -> None:
        """Initialize MixedScheduler."""
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
        num_batch = max(batch_counts) if self.fillup else min(batch_counts)

        return num_batch

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader], model: EmmentalModel = None
    ) -> Iterator[Union[Batch, List[Batch]]]:
        """Generate batch generator from all dataloaders in mixture for one epoch.

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
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        num_batch = max(batch_counts) if self.fillup else min(batch_counts)

        for batch_idx in range(num_batch):
            mixed_batch = []
            for (
                data_loader_idx,
                (task_to_label_dict, data_name, batch_count, split, uid_name),
            ) in enumerate(
                zip(task_to_label_dicts, data_names, batch_counts, splits, uid_names)
            ):
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

                mixed_batch.append(
                    Batch(
                        X_dict[uid_name],
                        X_dict,
                        Y_dict,
                        task_to_label_dict,
                        data_name,
                        split,
                    )
                )

            yield mixed_batch
