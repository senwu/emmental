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
        data_loaders = [[batch for batch in dataloader] for dataloader in dataloaders]

        dataloader_indexer = []
        for idx, count in enumerate(batch_counts):
            if self.fillup:
                dataloader_indexer.extend([idx] * max(batch_counts))
            else:
                dataloader_indexer.extend([idx] * count)

        random.shuffle(dataloader_indexer)

        batch_indexers = [0] * len(dataloaders)

        for index in dataloader_indexer:
            uid_name = uid_names[index]
            X_dict, Y_dict = data_loaders[index][
                batch_indexers[index] % batch_counts[index]
            ]
            batch_indexers[index] += 1
            yield X_dict[uid_name], X_dict, Y_dict, task_to_label_dicts[
                index
            ], data_names[index], splits[index]
