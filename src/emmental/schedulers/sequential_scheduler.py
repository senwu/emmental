from emmental.schedulers.scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Generate batch generator from all dataloaders in sequential order for MTL
    training.
    """

    def __init__(self, model, dataloaders):
        super().__init__(model, dataloaders)

    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in sequential order for
        one epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: A generator of all batches
        :rtype: genertor
        """

        task_names = [dataloader.task_name for dataloader in dataloaders]
        data_names = [dataloader.data_name for dataloader in dataloaders]
        label_names = [dataloader.label_name for dataloader in dataloaders]
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        for task_name, data_name, label_name, batch_count, data_loader in zip(
            task_names, data_names, label_names, batch_counts, data_loaders
        ):
            for batch in range(batch_count):
                yield task_name, data_name, label_name, next(data_loader)
