from typing import Dict, Iterator, List, Tuple, Union

from torch import Tensor

from emmental.data import EmmentalDataLoader
from emmental.schedulers.scheduler import Scheduler


class MixedScheduler(Scheduler):
    """Generate batch generator from all dataloaders in mixture for MTL training.

    :param fillup: Whether fillup to make all dataloader the same size
    :type fillup: bool
    """

    def __init__(self, fillup: bool = False) -> None:
        super().__init__()

        self.fillup = fillup

    def get_num_batches(self, dataloaders: List[EmmentalDataLoader]) -> int:
        """Get total number of batches per epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: Total number of batches per epoch
        :rtype: int
        """

        batch_counts = [len(dataloader) for dataloader in dataloaders]
        num_batch = max(batch_counts) if self.fillup else min(batch_counts)

        return num_batch

    def get_batches(
        self, dataloaders: List[EmmentalDataLoader]
    ) -> Iterator[
        List[
            Tuple[
                List[str],
                Dict[str, Union[Tensor, List[str]]],
                Dict[str, Tensor],
                Dict[str, str],
                str,
                str,
            ]
        ]
    ]:
        """Generate batch generator from all dataloaders in mixture for one epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: A generator of all batches
        :rtype: genertor
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
                    X_dict, Y_dict = next(data_loaders[data_loader_idx])
                except StopIteration:
                    data_loaders[data_loader_idx] = iter(dataloaders[data_loader_idx])
                    X_dict, Y_dict = next(data_loaders[data_loader_idx])

                mixed_batch.append(
                    (
                        X_dict[uid_name],
                        X_dict,
                        Y_dict,
                        task_to_label_dict,
                        data_name,
                        split,
                    )
                )

            yield mixed_batch
