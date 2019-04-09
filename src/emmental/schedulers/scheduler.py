from abc import ABC, abstractmethod


class Scheduler(ABC):
    """Generate batch generator from all dataloaders in designed order for MTL
    training.
    """

    @abstractmethod
    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in designed order for
        one epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        """

        pass
