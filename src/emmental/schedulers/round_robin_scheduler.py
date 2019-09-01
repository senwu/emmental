import random

from emmental.schedulers.scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    """Generate batch generator from all dataloaders in round robin order for MTL
    training.
    """

    def __init__(self, fillup=False):
        super().__init__()

        self.fillup = fillup

    def get_num_batches(self, dataloaders):
        """Get total number of batches per epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: Total number of batches per epoch
        :rtype: int
        """

        batch_counts = [len(dataloader) for dataloader in dataloaders]
        num_batch = (
            max(batch_counts) * len(dataloaders) if self.fillup else sum(batch_counts)
        )

        return num_batch

    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in round robin order for
        one epoch.

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

        dataloader_indexer = []
        for idx, count in enumerate(batch_counts):
            if self.fillup:
                dataloader_indexer.extend([idx] * max(batch_counts))
            else:
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
