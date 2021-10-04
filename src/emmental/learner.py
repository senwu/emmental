# Copyright (c) 2021 Sen Wu. All Rights Reserved.


"""Emmental learner."""
import collections
import copy
import importlib
import logging
import math
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import wandb
from numpy import ndarray
from torch import optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from emmental.data import EmmentalDataLoader
from emmental.logging import LoggingManager
from emmental.meta import Meta
from emmental.model import EmmentalModel
from emmental.optimizers.bert_adam import BertAdam
from emmental.schedulers import SCHEDULERS
from emmental.schedulers.scheduler import Scheduler
from emmental.utils.utils import construct_identifier, prob_to_pred

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmmentalLearner(object):
    """A class for emmental multi-task learning.

    Args:
      name: Name of the learner, defaults to None.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize EmmentalLearner."""
        self.name = name if name is not None else type(self).__name__

    def _set_logging_manager(self) -> None:
        """Set logging manager."""
        if Meta.config["learner_config"]["local_rank"] in [-1, 0]:
            if self.use_step_base_counter:
                self.logging_manager = LoggingManager(
                    self.n_batches_per_epoch, 0, self.start_step
                )
            else:
                self.logging_manager = LoggingManager(
                    self.n_batches_per_epoch,
                    self.start_epoch,
                    self.start_epoch * self.n_batches_per_epoch,
                )

    def _set_optimizer(self, model: EmmentalModel) -> None:
        """Set optimizer for learning process.

        Args:
          model: The model to set up the optimizer.
        """
        optimizer_config = Meta.config["learner_config"]["optimizer_config"]
        opt = optimizer_config["optimizer"]

        # If Meta.config["learner_config"]["optimizer_config"]["parameters"] is None,
        # create a parameter group with all parameters in the model, else load user
        # specified parameter groups.
        if optimizer_config["parameters"] is None:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
        else:
            parameters = optimizer_config["parameters"](model)

        optim_dict = {
            # PyTorch optimizer
            "asgd": optim.ASGD,
            "adadelta": optim.Adadelta,
            "adagrad": optim.Adagrad,
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "adamax": optim.Adamax,
            "lbfgs": optim.LBFGS,
            "rms_prop": optim.RMSprop,
            "r_prop": optim.Rprop,
            "sgd": optim.SGD,
            "sparse_adam": optim.SparseAdam,
            # Customized optimizer
            "bert_adam": BertAdam,
        }

        if opt in ["lbfgs", "r_prop", "sparse_adam"]:
            optimizer = optim_dict[opt](
                parameters,
                lr=optimizer_config["lr"],
                **optimizer_config[f"{opt}_config"],
            )
        elif opt in optim_dict.keys():
            optimizer = optim_dict[opt](
                parameters,
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["l2"],
                **optimizer_config[f"{opt}_config"],
            )
        elif (isinstance(opt, type) and issubclass(opt, optim.Optimizer)) or (
            isinstance(opt, partial)
            and issubclass(opt.func, optim.Optimizer)  # type: ignore
        ):
            optimizer = opt(parameters)  # type: ignore
        else:
            raise ValueError(f"Unrecognized optimizer option '{opt}'")

        self.optimizer = optimizer

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"Using optimizer {self.optimizer}")

        if Meta.config["learner_config"]["optimizer_path"]:
            try:
                self.optimizer.load_state_dict(
                    torch.load(
                        Meta.config["learner_config"]["optimizer_path"],
                        map_location=torch.device("cpu"),
                    )["optimizer"]
                )
                logger.info(
                    f"Optimizer state loaded from "
                    f"{Meta.config['learner_config']['optimizer_path']}"
                )
            except BaseException:
                logger.error(
                    f"Loading failed... Cannot load optimizer state from "
                    f"{Meta.config['learner_config']['optimizer_path']}, "
                    f"continuing anyway."
                )

    def _set_lr_scheduler(self, model: EmmentalModel) -> None:
        """Set learning rate scheduler for learning process.

        Args:
          model: The model to set up lr scheduler.
        """
        # Set warmup scheduler
        self._set_warmup_scheduler(model)

        # Set lr scheduler

        lr_scheduler_dict = {
            "exponential": optim.lr_scheduler.ExponentialLR,
            "plateau": optim.lr_scheduler.ReduceLROnPlateau,
            "step": optim.lr_scheduler.StepLR,
            "multi_step": optim.lr_scheduler.MultiStepLR,
            "cyclic": optim.lr_scheduler.CyclicLR,
            "one_cycle": optim.lr_scheduler.OneCycleLR,  # type: ignore
            "cosine_annealing": optim.lr_scheduler.CosineAnnealingLR,
        }

        opt = Meta.config["learner_config"]["lr_scheduler_config"]["lr_scheduler"]
        lr_scheduler_config = Meta.config["learner_config"]["lr_scheduler_config"]

        if opt is None:
            lr_scheduler = None
        elif opt == "linear":
            linear_decay_func = lambda x: (self.total_steps - self.warmup_steps - x) / (
                self.total_steps - self.warmup_steps
            )
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_decay_func
            )
        elif opt in ["exponential", "step", "multi_step", "cyclic"]:
            lr_scheduler = lr_scheduler_dict[opt](
                self.optimizer, **lr_scheduler_config[f"{opt}_config"]
            )
        elif opt == "one_cycle":
            lr_scheduler = lr_scheduler_dict[opt](
                self.optimizer,
                total_steps=self.total_steps,
                epochs=Meta.config["learner_config"]["n_epochs"]
                if not self.use_step_base_counter
                else 1,
                steps_per_epoch=self.n_batches_per_epoch
                if not self.use_step_base_counter
                else self.total_steps,
                **lr_scheduler_config[f"{opt}_config"],
            )
        elif opt == "cosine_annealing":
            lr_scheduler = lr_scheduler_dict[opt](
                self.optimizer,
                self.total_steps,
                eta_min=lr_scheduler_config["min_lr"],
                **lr_scheduler_config[f"{opt}_config"],
            )
        elif opt == "plateau":
            plateau_config = copy.deepcopy(lr_scheduler_config["plateau_config"])
            del plateau_config["metric"]
            lr_scheduler = lr_scheduler_dict[opt](
                self.optimizer,
                verbose=Meta.config["meta_config"]["verbose"],
                min_lr=lr_scheduler_config["min_lr"],
                **plateau_config,
            )
        elif isinstance(opt, _LRScheduler):
            lr_scheduler = opt(self.optimizer)  # type: ignore
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{opt}'")

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step_unit = Meta.config["learner_config"][
            "lr_scheduler_config"
        ]["lr_scheduler_step_unit"]
        self.lr_scheduler_step_freq = Meta.config["learner_config"][
            "lr_scheduler_config"
        ]["lr_scheduler_step_freq"]

        if Meta.config["meta_config"]["verbose"]:
            logger.info(
                f"Using lr_scheduler {repr(self.lr_scheduler)} with step every "
                f"{self.lr_scheduler_step_freq} {self.lr_scheduler_step_unit}."
            )

        if Meta.config["learner_config"]["scheduler_path"]:
            try:
                scheduler_state = torch.load(
                    Meta.config["learner_config"]["scheduler_path"]
                )["lr_scheduler"]
                if scheduler_state:
                    self.lr_scheduler.load_state_dict(scheduler_state)
                logger.info(
                    f"Lr scheduler state loaded from "
                    f"{Meta.config['learner_config']['scheduler_path']}"
                )
            except BaseException:
                logger.error(
                    f"Loading failed... Cannot load lr scheduler state from "
                    f"{Meta.config['learner_config']['scheduler_path']}, "
                    f"continuing anyway."
                )

    def _set_warmup_scheduler(self, model: EmmentalModel) -> None:
        """Set warmup learning rate scheduler for learning process.

        Args:
          model: The model to set up warmup scheduler.
        """
        self.warmup_steps = 0
        if Meta.config["learner_config"]["lr_scheduler_config"]["warmup_steps"]:
            warmup_steps = Meta.config["learner_config"]["lr_scheduler_config"][
                "warmup_steps"
            ]
            if warmup_steps < 0:
                raise ValueError("warmup_steps must greater than 0.")
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
            if Meta.config["meta_config"]["verbose"]:
                logger.info(f"Warmup {self.warmup_steps} batchs.")
        elif Meta.config["learner_config"]["lr_scheduler_config"]["warmup_percentage"]:
            warmup_percentage = Meta.config["learner_config"]["lr_scheduler_config"][
                "warmup_percentage"
            ]
            self.warmup_steps = math.ceil(warmup_percentage * self.total_steps)
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, linear_warmup_func
            )
            if Meta.config["meta_config"]["verbose"]:
                logger.info(f"Warmup {self.warmup_steps} batchs.")
        else:
            warmup_scheduler = None

        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(
        self, model: EmmentalModel, step: int, metric_dict: Dict[str, float]
    ) -> None:
        """Update the lr using lr_scheduler with each batch.

        Args:
          model: The model to update lr scheduler.
          step: The current step.
        """
        cur_lr = self.optimizer.param_groups[0]["lr"]

        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            lr_step_cnt = (
                self.lr_scheduler_step_freq
                if self.lr_scheduler_step_unit == "batch"
                else self.lr_scheduler_step_freq * self.n_batches_per_epoch
            )

            if (step + 1) % lr_step_cnt == 0:
                if (
                    Meta.config["learner_config"]["lr_scheduler_config"]["lr_scheduler"]
                    != "plateau"
                ):
                    self.lr_scheduler.step()
                elif (
                    Meta.config["learner_config"]["lr_scheduler_config"][
                        "plateau_config"
                    ]["metric"]
                    in metric_dict
                ):
                    self.lr_scheduler.step(
                        metric_dict[  # type: ignore
                            Meta.config["learner_config"]["lr_scheduler_config"][
                                "plateau_config"
                            ]["metric"]
                        ]
                    )

            min_lr = Meta.config["learner_config"]["lr_scheduler_config"]["min_lr"]
            if min_lr and self.optimizer.param_groups[0]["lr"] < min_lr:
                self.optimizer.param_groups[0]["lr"] = min_lr

        if (
            Meta.config["learner_config"]["lr_scheduler_config"]["reset_state"]
            and cur_lr != self.optimizer.param_groups[0]["lr"]
        ):
            logger.info("Reset the state of the optimizer.")
            self.optimizer.state = collections.defaultdict(dict)  # Reset state

    def _set_task_scheduler(self) -> None:
        """Set task scheduler for learning process."""
        opt = Meta.config["learner_config"]["task_scheduler_config"]["task_scheduler"]

        if opt in ["sequential", "round_robin", "mixed"]:
            self.task_scheduler = SCHEDULERS[opt](  # type: ignore
                **Meta.config["learner_config"]["task_scheduler_config"][
                    f"{opt}_scheduler_config"
                ]
            )
        elif isinstance(opt, Scheduler):
            self.task_scheduler = opt
        else:
            raise ValueError(f"Unrecognized task scheduler option '{opt}'")

    def _evaluate(
        self,
        model: EmmentalModel,
        dataloaders: List[EmmentalDataLoader],
        split: Union[List[str], str],
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
          model: The model to evaluate.
          dataloaders: The data to evaluate.
          split: The split to evaluate.

        Returns:
          The score dict.
        """
        if not isinstance(split, list):
            valid_split = [split]
        else:
            valid_split = split

        valid_dataloaders = [
            dataloader for dataloader in dataloaders if dataloader.split in valid_split
        ]
        return model.score(valid_dataloaders)

    def _logging(
        self,
        model: EmmentalModel,
        dataloaders: List[EmmentalDataLoader],
        batch_size: int,
    ) -> Dict[str, float]:
        """Check if it's time to evaluting or checkpointing.

        Args:
          model: The model to log.
          dataloaders: The data to evaluate.
          batch_size: Batch size.

        Returns:
          The score dict.
        """
        # Switch to eval mode for evaluation
        model.eval()

        metric_dict = dict()

        self.logging_manager.update(batch_size)

        trigger_evaluation = self.logging_manager.trigger_evaluation()

        # Log the loss and lr
        metric_dict.update(
            self._aggregate_running_metrics(
                model,
                trigger_evaluation and Meta.config["learner_config"]["online_eval"],
            )
        )

        # Evaluate the model and log the metric
        if trigger_evaluation:

            # Log task specific metric
            metric_dict.update(
                self._evaluate(
                    model, dataloaders, Meta.config["learner_config"]["valid_split"]
                )
            )

            self.logging_manager.write_log(metric_dict)

            self._reset_losses()
        elif Meta.config["logging_config"]["writer_config"]["write_loss_per_step"]:
            self.logging_manager.write_log(metric_dict)

        # Log metric dict every trigger evaluation time or full epoch
        if Meta.config["meta_config"]["verbose"] and (
            trigger_evaluation
            or self.logging_manager.epoch_total == int(self.logging_manager.epoch_total)
        ):
            logger.info(
                f"{self.logging_manager.counter_unit.capitalize()}: "
                f"{self.logging_manager.unit_total:.2f} {metric_dict}"
            )

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

    def _aggregate_running_metrics(
        self, model: EmmentalModel, calc_running_scores: bool = False
    ) -> Dict[str, float]:
        """Calculate the running overall and task specific metrics.

        Args:
          model: The model to evaluate.
          calc_running_scores: Whether to calc running scores

        Returns:
          The score dict.
        """
        metric_dict: Dict[str, float] = dict()

        total_count = 0
        # Log task specific loss
        for identifier in self.running_uids.keys():
            count = len(self.running_uids[identifier])
            if count > 0:
                metric_dict[identifier + "/loss"] = float(
                    self.running_losses[identifier] / count
                )
            total_count += count

        # Calculate average micro loss
        if total_count > 0:
            total_loss = sum(self.running_losses.values())
            metric_dict["model/all/train/loss"] = float(total_loss / total_count)

        if calc_running_scores:
            micro_score_dict: Dict[str, List[float]] = defaultdict(list)
            macro_score_dict: Dict[str, List[float]] = defaultdict(list)

            # Calculate training metric
            for identifier in self.running_uids.keys():
                task_name, data_name, split = identifier.split("/")

                metric_score = model.scorers[task_name].score(
                    self.running_golds[identifier],
                    self.running_probs[identifier],
                    prob_to_pred(self.running_probs[identifier]),
                    self.running_uids[identifier],
                )
                for metric_name, metric_value in metric_score.items():
                    metric_dict[f"{identifier}/{metric_name}"] = metric_value

                # Collect average score
                identifier = construct_identifier(
                    task_name, data_name, split, "average"
                )

                metric_dict[identifier] = np.mean(list(metric_score.values()))

                micro_score_dict[split].extend(
                    list(metric_score.values())  # type: ignore
                )
                macro_score_dict[split].append(metric_dict[identifier])

            # Collect split-wise micro/macro average score
            for split in micro_score_dict.keys():
                identifier = construct_identifier(
                    "model", "all", split, "micro_average"
                )
                metric_dict[identifier] = np.mean(
                    micro_score_dict[split]  # type: ignore
                )
                identifier = construct_identifier(
                    "model", "all", split, "macro_average"
                )
                metric_dict[identifier] = np.mean(
                    macro_score_dict[split]  # type: ignore
                )

        # Log the learning rate
        metric_dict["model/all/train/lr"] = self.optimizer.param_groups[0]["lr"]

        return metric_dict

    def _set_learning_counter(self) -> None:
        if Meta.config["learner_config"]["n_steps"]:
            if Meta.config["learner_config"]["skip_learned_data"]:
                self.start_epoch = 0
                self.start_step = 0
                self.start_train_epoch = 0
                self.start_train_step = Meta.config["learner_config"]["steps_learned"]
            else:
                self.start_epoch = 0
                self.start_step = Meta.config["learner_config"]["steps_learned"]
                self.start_train_epoch = 0
                self.start_train_step = Meta.config["learner_config"]["steps_learned"]
            self.end_epoch = 1
            self.end_step = Meta.config["learner_config"]["n_steps"]
            self.use_step_base_counter = True
            self.total_steps = Meta.config["learner_config"]["n_steps"]
        else:
            if Meta.config["learner_config"]["skip_learned_data"]:
                self.start_epoch = 0
                self.start_step = 0
                self.start_train_epoch = Meta.config["learner_config"]["epochs_learned"]
                self.start_train_step = Meta.config["learner_config"]["steps_learned"]
            else:
                self.start_epoch = Meta.config["learner_config"]["epochs_learned"]
                self.start_step = Meta.config["learner_config"]["steps_learned"]
                self.start_train_epoch = Meta.config["learner_config"]["epochs_learned"]
                self.start_train_step = Meta.config["learner_config"]["steps_learned"]
            self.end_epoch = Meta.config["learner_config"]["n_epochs"]
            self.end_step = self.n_batches_per_epoch
            self.use_step_base_counter = False
            self.total_steps = (
                Meta.config["learner_config"]["n_epochs"] * self.n_batches_per_epoch
            )

    def _reset_losses(self) -> None:
        """Reset running logs."""
        self.running_uids: Dict[str, List[str]] = defaultdict(list)
        self.running_losses: Dict[str, ndarray] = defaultdict(float)  # type: ignore
        self.running_probs: Dict[str, List[ndarray]] = defaultdict(list)
        self.running_golds: Dict[str, List[ndarray]] = defaultdict(list)

    def learn(
        self, model: EmmentalModel, dataloaders: List[EmmentalDataLoader]
    ) -> None:
        """Learning procedure of emmental MTL.

        Args:
          model: The emmental model that needs to learn.
          dataloaders: A list of dataloaders used to learn the model.
        """
        start_time = time.time()

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
                f'{Meta.config["learner_config"]["train_split"]} in dataloaders.'
            )

        # Set up task_scheduler
        self._set_task_scheduler()

        # Calculate the total number of batches per epoch
        self.n_batches_per_epoch: int = self.task_scheduler.get_num_batches(
            train_dataloaders
        )
        if self.n_batches_per_epoch == 0:
            logger.info("No batches in training dataloaders, existing...")
            return

        # Set up learning counter
        self._set_learning_counter()
        # Set up logging manager
        self._set_logging_manager()
        # Set up wandb watch model
        if (
            Meta.config["logging_config"]["writer_config"]["writer"] == "wandb"
            and Meta.config["logging_config"]["writer_config"]["wandb_watch_model"]
        ):
            if Meta.config["logging_config"]["writer_config"]["wandb_model_watch_freq"]:
                wandb.watch(
                    model,
                    log_freq=Meta.config["logging_config"]["writer_config"][
                        "wandb_model_watch_freq"
                    ],
                )
            else:
                wandb.watch(model)
        # Set up optimizer
        self._set_optimizer(model)
        # Set up lr_scheduler
        self._set_lr_scheduler(model)

        if Meta.config["learner_config"]["fp16"]:
            try:
                from apex import amp  # type: ignore
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to "
                    "use fp16 training."
                )
            logger.info(
                f"Modeling training with 16-bit (mixed) precision "
                f"and {Meta.config['learner_config']['fp16_opt_level']} opt level."
            )
            model, self.optimizer = amp.initialize(
                model,
                self.optimizer,
                opt_level=Meta.config["learner_config"]["fp16_opt_level"],
            )

        # Multi-gpu training (after apex fp16 initialization)
        if (
            Meta.config["learner_config"]["local_rank"] == -1
            and Meta.config["model_config"]["dataparallel"]
        ):
            model._to_dataparallel()

        # Distributed training (after apex fp16 initialization)
        if Meta.config["learner_config"]["local_rank"] != -1:
            model._to_distributed_dataparallel()

        # Set to training mode
        model.train()

        if Meta.config["meta_config"]["verbose"]:
            logger.info("Start learning...")

        self.metrics: Dict[str, float] = dict()
        self._reset_losses()

        # Set gradients of all model parameters to zero
        self.optimizer.zero_grad()

        batch_iterator = self.task_scheduler.get_batches(train_dataloaders, model)
        for epoch_num in range(self.start_epoch, self.end_epoch):
            for train_dataloader in train_dataloaders:
                # Set epoch for distributed sampler
                if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler
                ):
                    train_dataloader.sampler.set_epoch(epoch_num)
            step_pbar = tqdm(
                range(self.start_step, self.end_step),
                desc=f"Step {self.start_step + 1}/{self.end_step}"
                if self.use_step_base_counter
                else f"Epoch {epoch_num + 1}/{self.end_epoch}",
                disable=not Meta.config["meta_config"]["verbose"]
                or Meta.config["learner_config"]["local_rank"] not in [-1, 0],
            )
            for step_num in step_pbar:
                if self.use_step_base_counter:
                    step_pbar.set_description(f"Step {step_num + 1}/{self.total_steps}")
                    step_pbar.refresh()
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    batch_iterator = self.task_scheduler.get_batches(
                        train_dataloaders, model
                    )
                    batch = next(batch_iterator)

                # Check if skip the current batch
                if epoch_num < self.start_train_epoch or (
                    epoch_num == self.start_train_epoch
                    and step_num < self.start_train_step
                ):
                    continue

                # Covert single batch into a batch list
                if not isinstance(batch, list):
                    batch = [batch]

                total_step_num = epoch_num * self.n_batches_per_epoch + step_num
                batch_size = 0

                for uids, X_dict, Y_dict, task_to_label_dict, data_name, split in batch:
                    batch_size += len(uids)

                    # Perform forward pass and calcualte the loss and count
                    uid_dict, loss_dict, prob_dict, gold_dict = model(
                        uids,
                        X_dict,
                        Y_dict,
                        task_to_label_dict,
                        return_probs=Meta.config["learner_config"]["online_eval"],
                        return_action_outputs=False,
                    )

                    # Update running loss and count
                    for task_name in uid_dict.keys():
                        identifier = f"{task_name}/{data_name}/{split}"
                        self.running_uids[identifier].extend(uid_dict[task_name])
                        self.running_losses[identifier] += (
                            loss_dict[task_name].item() * len(uid_dict[task_name])
                            if len(loss_dict[task_name].size()) == 0
                            else torch.sum(loss_dict[task_name]).item()
                        ) * model.task_weights[task_name]
                        if Meta.config["learner_config"]["online_eval"]:
                            self.running_probs[identifier].extend(prob_dict[task_name])
                            self.running_golds[identifier].extend(gold_dict[task_name])

                    # Calculate the average loss
                    loss = sum(
                        [
                            model.task_weights[task_name] * task_loss
                            if len(task_loss.size()) == 0
                            else torch.mean(model.task_weights[task_name] * task_loss)
                            for task_name, task_loss in loss_dict.items()
                        ]
                    )

                    # Perform backward pass to calculate gradients
                    if Meta.config["learner_config"]["fp16"]:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()  # type: ignore

                if (total_step_num + 1) % Meta.config["learner_config"][
                    "optimizer_config"
                ]["gradient_accumulation_steps"] == 0 or (
                    step_num + 1 == self.end_step and epoch_num + 1 == self.end_epoch
                ):
                    # Clip gradient norm
                    if Meta.config["learner_config"]["optimizer_config"]["grad_clip"]:
                        if Meta.config["learner_config"]["fp16"]:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                Meta.config["learner_config"]["optimizer_config"][
                                    "grad_clip"
                                ],
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                Meta.config["learner_config"]["optimizer_config"][
                                    "grad_clip"
                                ],
                            )

                    # Update the parameters
                    self.optimizer.step()

                    # Set gradients of all model parameters to zero
                    self.optimizer.zero_grad()

                if Meta.config["learner_config"]["local_rank"] in [-1, 0]:
                    self.metrics.update(self._logging(model, dataloaders, batch_size))

                    step_pbar.set_postfix(self.metrics)

                # Update lr using lr scheduler
                self._update_lr_scheduler(model, total_step_num, self.metrics)
            step_pbar.close()

        if Meta.config["learner_config"]["local_rank"] in [-1, 0]:
            model = self.logging_manager.close(model)
        logger.info(f"Total learning time: {time.time() - start_time} seconds.")
