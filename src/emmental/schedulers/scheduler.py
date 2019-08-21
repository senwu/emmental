from abc import ABC, abstractmethod


class Scheduler(ABC):
    """Generate batch generator from all dataloaders in designed order for MTL
    training.
    """

    def __init__(self):

        pass

    def get_num_batches(dataloaders):
        """Get total number of batches per epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: Total number of batches per epoch
        :rtype: int
        """

        raise NotImplementedError()

    @abstractmethod
    def get_batches(self, dataloaders):
        """Generate batch generator from all dataloaders in designed order for
        one epoch.

        :param dataloaders: a list of dataloaders
        :type dataloaders: list
        :return: A generator of all batches
        :rtype: genertor
        """

        pass
