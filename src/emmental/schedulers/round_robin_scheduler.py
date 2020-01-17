import random
from typing import Dict, Iterator, List, Tuple, Union

from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.model import EmmentalModel
from emmental.schedulers.scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    r"""Generate batch generator from all dataloaders in round robin order for MTL
      training.

    Args:
      fillup(bool): Whether fillup to make all dataloader the same size.

    """

    def __init__(self, fillup: bool = False) -> None:
        super().__init__()

        self.fillup = fillup

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        r"""Get total number of batches per epoch.

        Args:
          dataloaders(list): List of dataloaders.

        Returns:
          int: Total number of batches per epoch.

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
        r"""Generate batch generator from all dataloaders in round robin order for
          one epoch.

        Args:
          dataloaders(list): List of dataloaders.
          model(EmmentalModel): The training model, defaults to None.

        Returns:
          genertor: A generator of all batches.

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
                X_dict, Y_dict = next(data_loaders[data_loader_idx])
            except StopIteration:
                data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                X_dict, Y_dict = next(data_loaders[data_loader_idx])

            yield X_dict[uid_name], X_dict, Y_dict, task_to_label_dicts[
                data_loader_idx
            ], data_names[data_loader_idx], splits[data_loader_idx]
