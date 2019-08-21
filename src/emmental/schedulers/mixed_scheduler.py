from emmental.schedulers.scheduler import Scheduler


class MixedScheduler(Scheduler):
    """Generate batch generator from all dataloaders in mixture for MTL training.
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
        num_batch = max(batch_counts) if self.fillup else min(batch_counts)

        return num_batch

    def get_batches(self, dataloaders):
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
        data_loaders = [[batch for batch in dataloader] for dataloader in dataloaders]

        num_batch = max(batch_counts) if self.fillup else min(batch_counts)

        for batch_idx in range(num_batch):
            mixed_batch = []
            for (
                task_to_label_dict,
                data_name,
                batch_count,
                data_loader,
                split,
                uid_name,
            ) in zip(
                task_to_label_dicts,
                data_names,
                batch_counts,
                data_loaders,
                splits,
                uid_names,
            ):
                X_dict, Y_dict = data_loader[batch_idx % batch_count]
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
