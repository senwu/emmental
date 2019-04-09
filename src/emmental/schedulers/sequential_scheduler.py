from emmental.schedulers.scheduler import Scheduler


class SequentialScheduler(Scheduler):
    """Generate batch generator from all dataloaders in sequential order for MTL
    training.
    """

    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in sequential order for
        one epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: A generator of all batches
        :rtype: genertor
        """

        task_names = [dataloader.name for dataloader in dataloaders]
        batch_counts = [len(dataloader) for dataloader in dataloaders]
        data_loaders = [iter(dataloader) for dataloader in dataloaders]

        for task_name, batch_count, data_loader in zip(
            task_names, batch_counts, data_loaders
        ):
            for batch in batch_count:
                yield task_name, next(data_loader)
