from emmental.schedulers.scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Generate batch generator from all dataloaders in sequential order for MTL
    training.
    """

    def __init__(self):
        super().__init__()

    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in sequential order for
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

        for task_to_label_dict, data_name, batch_count, data_loader, split in zip(
            task_to_label_dicts, data_names, batch_counts, data_loaders, splits
        ):
            for batch in range(batch_count):
                yield next(data_loader), task_to_label_dict, data_name, split
