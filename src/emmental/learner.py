import numpy as np
import torch
import torch.optim as optim

from emmental.schedulers.sequential_scheduler import SequentialScheduler
from emmental.utils.config import _merge
from emmental.utils.logging import Counter, LogWriter, TensorBoardWriter

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


class EmmentalLearner(object):
    """A class for emmental multi-task learning.

    :param config: The learning config
    :type config: dict
    """

    def __init__(self, config):
        self.config = config

        # Set random seed for learning
        if "seed" not in self.config:
            self.config["seed"] = np.random.randint(1e5)
        self._set_random_seed(self.config["seed"])

    def _set_random_seed(self, seed):
        """Set random seed."""

        # Set random seed for all numpy operations
        self.rand_state = np.random.RandomState(seed=seed)
        np.random.seed(seed=seed)

        # Set random seed for PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _set_writer(self):
        """Set learning log writer."""

        writer_config = self.config["writer_config"]
        opt = writer_config["writer"]

        if opt is None:
            self.writer = None
        elif opt == "json":
            self.writer = LogWriter()
        elif opt == "tensorboard":
            self.writer = TensorBoardWriter()
        else:
            raise ValueError(f"Unrecognized writer option '{opt}'")

    def _set_counter(self):
        """Set counter for learning process."""

        self.counter = Counter(self.config, self.n_batches_per_epoch)

    def _set_optimizer(self, model):
        """Set optimizer for learning process."""

        # TODO: add more optimizer support
        optimizer_config = self.config["learner_config"]["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config["sgd_config"],
                weight_decay=optimizer_config["l2"],
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config["adam_config"],
                weight_decay=optimizer_config["l2"],
            )
        else:
            raise ValueError(f"Unrecognized optimizer option '{opt}'")

        self.optimizer = optimizer

    def _set_lr_scheduler(self, model):
        """Set learning rate scheduler for learning process."""

        # TODO: add more lr scheduler support
        opt = self.config["learner_config"]["lr_scheduler_config"]["lr_scheduler"]
        lr_scheduler_config = self.config["learner_config"]["lr_scheduler_config"]

        if opt is None:
            lr_scheduler = None
        elif opt == "exponential":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, **lr_scheduler_config["exponential_config"]
            )
        elif opt == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **lr_scheduler_config["step_config"]
            )
        elif opt == "multi_step":
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, **lr_scheduler_config["multi_step_config"]
            )
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{opt}'")

        self.lr_scheduler = lr_scheduler

    def _update_lr_scheduler(self, model, step):
        """Update the lr using lr_scheduler with each batch."""

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _set_task_scheduler(self, model, dataloaders):
        """Set task scheduler for learning process"""
        # TODO: add more task scheduler support
        opt = self.config["learner_config"]["task_scheduler"]

        if opt == "sequential":
            self.task_scheduler = SequentialScheduler(model, dataloaders)
        else:
            raise ValueError(f"Unrecognized task scheduler option '{opt}'")

    def _evaluating_checkpointing(self, model, dataloaders, batch_size):
        """Checking if it's time to evaluting or checkpointing"""

        model.eval()
        metric_dict = dict()

        metric_dict.update(self.calculate_metrics(model, dataloaders, split="train"))

        metric_dict.update(
            self.model.calculate_metrics(model, dataloaders, split="train")
        )
        # self.logger.update()

        # if self.logger.trigger_evaluation():

        # if self.logger.trigger_checkpointing():

        return metric_dict

    def learn(self, model, dataloaders, config={}):
        """The learning procedure of emmental MTL

        :param model: The emmental model that needs to learn
        :type model: emmental.model
        :param dataloaders: a list of dataloaders used to learn the model
        :type dataloaders: list
        :param config: the config to update the existing config, defaults to {}
        :type config: dict, optional
        """

        # Update the existing config
        self.config = _merge(self.config, config)

        # Generate the list of dataloaders for learning process
        train_dataloaders = [
            dataloader
            for dataloader in dataloaders
            if dataloader.split == self.config["learner_config"]["train_split"]
        ]

        # Calculate the total number of batches per epoch
        self.n_batches_per_epoch = sum(
            [len(dataloader) for dataloader in train_dataloaders]
        )

        # Set up counter
        self._set_counter()
        # Set up optimizer
        self._set_optimizer(model)
        # Set up lr_scheduler
        self._set_lr_scheduler(model)
        # Set up task_scheduler
        self._set_task_scheduler(model, dataloaders)
        # Set up writer
        self._set_writer()

        # Set to training mode
        model.train()

        for epoch in range(self.config["learner_config"]["n_epochs"]):
            batches = tqdm(
                enumerate(self.task_scheduler.get_batches(train_dataloaders)),
                total=self.n_batches_per_epoch,
                disable=(
                    not (
                        self.config["learner_config"]["progress_bar"]
                        and self.config["learner_config"]["verbose"]
                    )
                ),
            )
            for batch_num, (task_name, data_name, label_name, batch) in batches:
                total_batch_num = epoch * self.n_batches_per_epoch + batch_num

                X_dict, Y_dict = batch

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict, count_dict = model.calculate_losses(
                    X_dict["data"], Y_dict, [task_name], [label_name]
                )

                # Calculate the average loss
                loss = sum(loss_dict.values())
                # print(task_name, loss.item())
                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm
                if self.config["learner_config"]["optimizer_config"]["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config["learner_config"]["optimizer_config"]["grad_clip"],
                    )

                # Update the parameters
                self.optimizer.step()

                # Update lr using lr scheduler
                self._update_lr_scheduler(model, total_batch_num)

                # metric_dict = self._evaluate_checkpoint(model, dataloaders)

                # batches.set_postfix(loss_dict)
        # pass
