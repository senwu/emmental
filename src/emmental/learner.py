import logging
from collections import defaultdict

import torch
import torch.optim as optim

from emmental import Meta
from emmental.logging import LoggingManager
from emmental.schedulers.round_robin_scheduler import RoundRobinScheduler
from emmental.schedulers.sequential_scheduler import SequentialScheduler

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

    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__

    def _set_logging_manager(self):
        """Set logging manager."""

        self.logging_manager = LoggingManager(self.n_batches_per_epoch)

    def _set_optimizer(self, model):
        """Set optimizer for learning process."""

        # TODO: add more optimizer support and fp16
        optimizer_config = Meta.config["learner_config"]["optimizer_config"]
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
        opt = Meta.config["learner_config"]["lr_scheduler_config"]["lr_scheduler"]
        lr_scheduler_config = Meta.config["learner_config"]["lr_scheduler_config"]

        if opt is None:
            lr_scheduler = None
        elif opt == "linear":
            total_steps = (
                self.n_batches_per_epoch * Meta.config["learner_config"]["n_epochs"]
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
        elif opt == "reduce_on_plateau":
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

        if Meta.config["learner_config"]["lr_scheduler_config"]["warmup_steps"]:
            warmup_steps = Meta.config["learner_config"]["lr_scheduler_config"][
                "warmup_steps"
            ]
            if warmup_steps < 0:
                raise ValueError(f"warmup_steps much greater or equal than 0.")
            warmup_unit = Meta.config["learner_config"]["lr_scheduler_config"][
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
            logger.info(f"Warmup {self.warmup_steps} batchs.")
        elif Meta.config["learner_config"]["lr_scheduler_config"]["warmup_percentage"]:
            warmup_percentage = Meta.config["learner_config"]["lr_scheduler_config"][
                "warmup_percentage"
            ]
            self.warmup_steps = int(
                warmup_percentage
                * Meta.config["learner_config"]["n_epochs"]
                * self.n_batches_per_epoch
            )
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_warmup_func
            )
            logger.info(f"Warmup {self.warmup_steps} batchs.")
        else:
            warmup_scheduler = None
            self.warmup_steps = 0

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, model, step):
        """Update the lr using lr_scheduler with each batch."""

        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()
            min_lr = Meta.config["learner_config"]["lr_scheduler_config"]["min_lr"]
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

    def _set_task_scheduler(self, model, dataloaders):
        """Set task scheduler for learning process"""
        # TODO: add more task scheduler support
        opt = Meta.config["learner_config"]["task_scheduler"]

        if opt == "sequential":
            self.task_scheduler = SequentialScheduler()
        elif opt == "round_robin":
            self.task_scheduler = RoundRobinScheduler()
        else:
            raise ValueError(f"Unrecognized task scheduler option '{opt}'")

    def _evaluate(self, model, dataloaders, split):
        if not isinstance(split, list):
            valid_split = [split]
        else:
            valid_split = split

        valid_dataloaders = [
            dataloader for dataloader in dataloaders if dataloader.split in valid_split
        ]
        return model.score(valid_dataloaders)

    def _logging(self, model, dataloaders, batch_size):
        """Checking if it's time to evaluting or checkpointing"""

        # Switch to eval mode for evaluation
        model.eval()

        metric_dict = dict()

        self.logging_manager.update(batch_size)

        # Log the loss and lr
        metric_dict.update(self._aggregate_losses())

        # Evaluate the model and log the metric
        if self.logging_manager.trigger_evaluation():

            # Log task specific metric
            metric_dict.update(
                self._evaluate(
                    model, dataloaders, Meta.config["learner_config"]["valid_split"]
                )
            )

            self.logging_manager.write_log(metric_dict)

            self._reset_losses()

        # Checkpoint the model
        if self.logging_manager.trigger_checkpointing():
            self.logging_manager.checkpoint_model(
                model, self.optimizer, self.lr_scheduler, metric_dict
            )

            self.logging_manager.write_log(metric_dict)

            self._reset_losses()

        # Switch to train mode
        model.train()

        return metric_dict

    def _aggregate_losses(self):
        """Calculate the task specific loss, average micro loss and learning rate."""

        metric_dict = dict()

        # Log task specific loss
        for identifier in self.running_losses.keys():
            if self.running_counts[identifier] > 0:
                metric_dict[identifier] = (
                    self.running_losses[identifier] / self.running_counts[identifier]
                )

        # Calculate average micro loss
        total_loss = sum(self.running_losses.values())
        total_count = sum(self.running_counts.values())
        if total_count > 0:
            metric_dict["model/train/all/loss"] = total_loss / total_count

        # Log the learning rate
        metric_dict["model/train/all/lr"] = self.optimizer.param_groups[0]["lr"]

        return metric_dict

    def _reset_losses(self):
        self.running_losses = defaultdict(float)
        self.running_counts = defaultdict(int)

    def learn(self, model, dataloaders):
        """The learning procedure of emmental MTL

        :param model: The emmental model that needs to learn
        :type model: emmental.model
        :param dataloaders: a list of dataloaders used to learn the model
        :type dataloaders: list
        """

        # Generate the list of dataloaders for learning process
        train_split = Meta.config["learner_config"]["train_split"]
        if isinstance(train_split, str):
            train_split = [train_split]

        train_dataloaders = [
            dataloader for dataloader in dataloaders if dataloader.split in train_split
        ]

        if not train_dataloaders:
            raise ValueError(
                f"Cannot find the specified train_split "
                f'{Meta.config["learner_config"]["train_split"]} in datloaders.'
            )

        # Calculate the total number of batches per epoch
        self.n_batches_per_epoch = sum(
            [len(dataloader) for dataloader in train_dataloaders]
        )

        # Set up logging manager
        self._set_logging_manager()
        # Set up optimizer
        self._set_optimizer(model)
        # Set up lr_scheduler
        self._set_lr_scheduler(model)
        # Set up task_scheduler
        self._set_task_scheduler(model, dataloaders)

        # Set to training mode
        model.train()

        logger.info(f"Start learning...")

        self.metrics = dict()
        self._reset_losses()

        for epoch_num in range(Meta.config["learner_config"]["n_epochs"]):
            batches = tqdm(
                enumerate(self.task_scheduler.get_batches(train_dataloaders)),
                total=self.n_batches_per_epoch,
                disable=(not Meta.config["meta_config"]["verbose"]),
                desc=f"Epoch {epoch_num}:",
            )
            for batch_num, (batch, task_to_label_dict, data_name, split) in batches:
                X_dict, Y_dict = batch

                total_batch_num = epoch_num * self.n_batches_per_epoch + batch_num
                batch_size = len(next(iter(Y_dict.values())))

                # Update lr using lr scheduler
                self._update_lr_scheduler(model, total_batch_num)

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict, count_dict = model.calculate_loss(
                    X_dict, Y_dict, task_to_label_dict, data_name, split
                )

                # Update running loss and count
                for identifier in loss_dict.keys():
                    self.running_losses[identifier] += (
                        loss_dict[identifier].item() * count_dict[identifier]
                    )
                    self.running_counts[identifier] += count_dict[identifier]

                # Skip the backward pass if no loss is calcuated
                if not loss_dict:
                    continue

                # Calculate the average loss
                loss = sum(loss_dict.values())

                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm
                if Meta.config["learner_config"]["optimizer_config"]["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        Meta.config["learner_config"]["optimizer_config"]["grad_clip"],
                    )

                # Update the parameters
                self.optimizer.step()

                self.metrics.update(self._logging(model, dataloaders, batch_size))

                batches.set_postfix(self.metrics)

        model = self.logging_manager.close(model)
