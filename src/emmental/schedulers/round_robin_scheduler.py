import random

from emmental.schedulers.scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    """Generate batch generator from all dataloaders in round robin order for MTL
    training.
    """

    def __init__(self):
        super().__init__()

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
        data_names = [dataloader.data_name for dataloader in dataloaders]
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]
        splits = [dataloader.split for dataloader in dataloaders]

        dataloader_indexer = []
        for idx, count in enumerate(batch_counts):
            dataloader_indexer.extend([idx] * count)

        random.shuffle(dataloader_indexer)

        for index in dataloader_indexer:
            yield next(data_loaders[index]), task_to_label_dicts[index], data_names[
                index
            ], splits[index]
