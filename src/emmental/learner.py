import logging

import numpy as np
import torch
import torch.optim as optim

from emmental.schedulers.sequential_scheduler import SequentialScheduler
from emmental.utils.config import _merge
from emmental.utils.logging import LoggingManager
from emmental.utils.utils import set_random_seed

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm

logger = logging.getLogger(__name__)


class EmmentalLearner(object):
    """A class for emmental multi-task learning.

    :param config: The learning config
    :type config: dict
    """

    def __init__(self, config):
        # Set the config
        self.config = config

        # Set random seed for learning
        if "seed" not in self.config:
            self.config["seed"] = np.random.randint(1e5)
        set_random_seed(self.config["seed"])

    # def _set_writer(self):
    #     """Set learning log writer."""

    #     writer_config = self.config["writer_config"]
    #     opt = writer_config["writer"]

    #     if opt is None:
    #         self.writer = None
    #     elif opt == "json":
    #         self.writer = LogWriter()
    #     elif opt == "tensorboard":
    #         self.writer = TensorBoardWriter()
    #     else:
    #         raise ValueError(f"Unrecognized writer option '{opt}'")

    def _set_logging_manager(self):
        """Set logging manager."""

        self.logging_manager = LoggingManager(self.config, self.n_batches_per_epoch)

    # def _set_counter(self):
    #     """Set counter for learning process."""

    #     self.counter = Counter(self.config, self.n_batches_per_epoch)

    def _set_optimizer(self, model):
        """Set optimizer for learning process."""

        # TODO: add more optimizer support and fp16
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

        # Set warmup scheduler
        self._set_warmup_scheduler(model)

        # Set lr scheduler
        # TODO: add more lr scheduler support
        opt = self.config["learner_config"]["lr_scheduler_config"]["lr_scheduler"]
        lr_scheduler_config = self.config["learner_config"]["lr_scheduler_config"]

        if opt is None:
            lr_scheduler = None
        elif opt == "linear":
            total_steps = (
                self.n_batches_per_epoch * self.config["learner_config"]["n_epochs"]
            )
            linear_decay_func = lambda x: (total_steps - self.warmup_steps - x) / (
                total_steps - self.warmup_steps
            )
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_decay_func
            )
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
        elif lr_scheduler == "reduce_on_plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                min_lr=lr_scheduler_config["min_lr"],
                **lr_scheduler_config["plateau_config"],
            )
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{opt}'")

        self.lr_scheduler = lr_scheduler

    def _set_warmup_scheduler(self, model):
        """Set warmup learning rate scheduler for learning process."""

        if self.config["learner_config"]["lr_scheduler_config"]["warmup_steps"]:
            warmup_steps = self.config["learner_config"]["lr_scheduler_config"][
                "warmup_steps"
            ]
            if warmup_steps < 0:
                raise ValueError(f"warmup_steps much greater or equal than 0.")
            warmup_unit = self.config["learner_config"]["lr_scheduler_config"][
                "warmup_unit"
            ]
            if warmup_unit == "epoch":
                self.warmup_steps = int(warmup_steps * self.n_batches_per_epoch)
            elif warmup_unit == "batch":
                self.warmup_steps = int(warmup_steps)
            else:
                raise ValueError(
                    f"warmup_unit must be 'batch' or 'epoch', but {warmup_unit} found."
                )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_warmup_func
            )
        else:
            warmup_scheduler = None
            self.warmup_step = 0

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, model, step):
        """Update the lr using lr_scheduler with each batch."""

        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()
            min_lr = self.config["learner_config"]["lr_scheduler_config"]["min_lr"]
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

    def _set_task_scheduler(self, model, dataloaders):
        """Set task scheduler for learning process"""
        # TODO: add more task scheduler support
        opt = self.config["learner_config"]["task_scheduler"]

        if opt == "sequential":
            self.task_scheduler = SequentialScheduler()
        else:
            raise ValueError(f"Unrecognized task scheduler option '{opt}'")

    def _evaluate(self, model, dataloaders, split):
        valid_dataloaders = [
            dataloader for dataloader in dataloaders if dataloader.split == split
        ]
        return model.score(valid_dataloaders)

    def _logging(self, model, dataloaders, batch_size):
        """Checking if it's time to evaluting or checkpointing"""

        model.eval()
        metric_dict = dict()

        self.logging_manager.update(batch_size)
        # print(self.counter.unit_total, self.counter.trigger_evaluation())
        if self.logging_manager.trigger_evaluation():
            metric_dict.update(
                self._evaluate(
                    model, dataloaders, self.config["learner_config"]["valid_split"]
                )
            )

            self.logging_manager.write_log(metric_dict)
            # for metric_name, metric_value in metric_dict.items():
            #     print(metric_name, metric_value, self.logging_manager.unit_total)
            #     self.writer.add_scalar(
            #         metric_name, metric_value, self.counter.unit_total
            #     )
            #     self.writer.add_scalar("model/loss",
            # self.optimizer.param_groups[0]["lr"],
            #  self.counter.unit_total)

        if self.logging_manager.trigger_checkpointing():
            self.logging_manager.checkpoint_model(
                model, self.optimizer, self.lr_scheduler, metric_dict
            )

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

        if not train_dataloaders:
            raise ValueError(
                f"Cannot find the specified train_split "
                f'{self.config["learner_config"]["train_split"]} in datloaders.'
            )

        # Calculate the total number of batches per epoch
        self.n_batches_per_epoch = sum(
            [len(dataloader) for dataloader in train_dataloaders]
        )

        # Set up logging manager
        self._set_logging_manager()
        # # Set up counter
        # self._set_counter()
        # Set up optimizer
        self._set_optimizer(model)
        # Set up lr_scheduler
        self._set_lr_scheduler(model)
        # Set up task_scheduler
        self._set_task_scheduler(model, dataloaders)
        # # Set up writer
        # self._set_writer()

        # Set to training mode
        model.train()

        logger.info(f"Start learning...")

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
                X_dict, Y_dict = batch

                total_batch_num = epoch * self.n_batches_per_epoch + batch_num
                batch_size = len(next(iter(Y_dict.values())))

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict, count_dict = model.calculate_losses(
                    X_dict,
                    Y_dict,
                    [task_name],
                    [data_name],
                    [label_name],
                    self.config["learner_config"]["train_split"],
                )

                # Skip the backward pass if no loss is calcuated
                if not loss_dict:
                    continue

                # Calculate the average loss
                loss = sum(loss_dict.values())

                # Perform backward pass to calculate gradients
                loss.backward()

                # self.logging_manager.write_log(loss_dict)
                # for loss_name, loss_value in loss_dict.items():
                #     self.writer.add_scalar(
                #         loss_name, loss_value.item(), total_batch_num
                #     )

                # Clip gradient norm
                if self.config["learner_config"]["optimizer_config"]["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config["learner_config"]["optimizer_config"]["grad_clip"],
                    )

                # Update the parameters
                self.optimizer.step()

                metrics_dict = self._logging(model, dataloaders, batch_size)

                self.logging_manager.write_log(loss_dict)

                # print(metrics_dict)
                # Update lr using lr scheduler
                self._update_lr_scheduler(model, total_batch_num)

                batches.set_postfix(metrics_dict)
