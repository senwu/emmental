import argparse
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional

from emmental.utils.utils import (
    nullable_float,
    nullable_int,
    nullable_string,
    str2bool,
    str2dict,
)


def parse_arg(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    r"""Parse the configuration from command line.

    Args:
      parser(ArgumentParser): The exterenl argument parser object, defaults to None.

    Returns:
      ArgumentParser: The updated argument parser object.

    """

    if parser is None:
        parser = argparse.ArgumentParser(
            "Emmental configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    # Load meta configuration
    meta_config = parser.add_argument_group("Meta configuration")

    meta_config.add_argument(
        "--seed",
        type=nullable_int,
        default=0,
        help="Random seed for all numpy/torch/cuda operations in model and learning",
    )

    meta_config.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Whether to print the log information",
    )

    meta_config.add_argument(
        "--log_path", type=str, default="logs", help="Directory to save running log"
    )

    # Load data configuration
    data_config = parser.add_argument_group("Data configuration")

    data_config.add_argument(
        "--min_data_len", type=int, default=0, help="Minimal data length"
    )

    data_config.add_argument(
        "--max_data_len",
        type=int,
        default=0,
        help="Maximal data length (0 for no max_len)",
    )

    # Load model configuration
    model_config = parser.add_argument_group("Model configuration")

    model_config.add_argument(
        "--model_path",
        type=nullable_string,
        default=None,
        help="Path to pretrained model",
    )

    model_config.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which device to use (-1 for cpu or gpu id (e.g., 0 for cuda:0))",
    )

    model_config.add_argument(
        "--dataparallel",
        type=str2bool,
        default=True,
        help="Whether to use dataparallel or not",
    )

    # Learning configuration
    learner_config = parser.add_argument_group("Learning configuration")

    learner_config.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision to train",
    )

    learner_config.add_argument(
        "--n_epochs", type=int, default=3, help="Total number of learning epochs"
    )

    learner_config.add_argument(
        "--train_split", type=str, default="train", help="The split for training"
    )

    learner_config.add_argument(
        "--valid_split",
        nargs="+",
        type=str,
        default=["valid"],
        help="The split for validation",
    )

    learner_config.add_argument(
        "--test_split", type=str, default="test", help="The split for testing"
    )

    learner_config.add_argument(
        "--ignore_index",
        type=nullable_int,
        default=None,
        help="The ignore index, uses for masking samples",
    )

    # Optimizer configuration
    optimizer_config = parser.add_argument_group("Optimizer configuration")

    optimizer_config.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamax", "sgd", "bert_adam"],
        help="The optimizer to use",
    )

    optimizer_config.add_argument("--lr", type=float, default=1e-3, help="Learing rate")

    optimizer_config.add_argument(
        "--l2", type=float, default=0.0, help="l2 regularization"
    )

    optimizer_config.add_argument(
        "--grad_clip", type=nullable_float, default=None, help="Gradient clipping"
    )

    optimizer_config.add_argument(
        "--sgd_momentum", type=float, default=0.9, help="SGD momentum"
    )

    optimizer_config.add_argument(
        "--sgd_dampening", type=float, default=0, help="SGD dampening"
    )

    optimizer_config.add_argument(
        "--sgd_nesterov", type=str2bool, default=False, help="SGD nesterov"
    )

    # TODO: add adam/adamax/bert_adam betas

    optimizer_config.add_argument(
        "--amsgrad",
        type=str2bool,
        default=False,
        help="Whether to use the AMSGrad variant of adam",
    )

    optimizer_config.add_argument(
        "--eps", type=float, default=1e-8, help="eps in adam, adamax, or bert_adam"
    )

    # Scheduler configuration
    scheduler_config = parser.add_argument_group("Scheduler configuration")

    scheduler_config.add_argument(
        "--lr_scheduler",
        type=nullable_string,
        default=None,
        choices=["linear", "exponential", "step", "multi_step"],
        help="Learning rate scheduler",
    )

    scheduler_config.add_argument(
        "--warmup_steps", type=float, default=None, help="Warm up steps"
    )

    scheduler_config.add_argument(
        "--warmup_unit",
        type=str,
        default="batch",
        choices=["epoch", "batch"],
        help="Warm up unit",
    )

    scheduler_config.add_argument(
        "--warmup_percentage", type=float, default=None, help="Warm up percentage"
    )

    scheduler_config.add_argument(
        "--min_lr", type=float, default=0.0, help="Minimum learning rate"
    )

    scheduler_config.add_argument(
        "--linear_lr_scheduler_min_lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for linear lr scheduler",
    )

    scheduler_config.add_argument(
        "--exponential_lr_scheduler_gamma",
        type=float,
        default=0.9,
        help="Gamma for exponential lr scheduler",
    )

    scheduler_config.add_argument(
        "--plateau_lr_scheduler_factor",
        type=float,
        default=0.5,
        help="factor for plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--plateau_lr_scheduler_patience",
        type=int,
        default=10,
        help="Patience for plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--plateau_lr_scheduler_threshold",
        type=float,
        default=0.0001,
        help="Threshold for plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--step_lr_scheduler_step_size",
        type=int,
        default=1,
        help="Period of learning rate decay",
    )

    scheduler_config.add_argument(
        "--step_lr_scheduler_gamma",
        type=float,
        default=0.01,
        help="Multiplicative factor of learning rate decay",
    )

    scheduler_config.add_argument(
        "--step_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
    )

    scheduler_config.add_argument(
        "--multi_step_lr_scheduler_milestones",
        nargs="+",
        type=int,
        default=[10000],
        help="List of epoch indices. Must be increasing.",
    )

    scheduler_config.add_argument(
        "--multi_step_lr_scheduler_gamma",
        type=float,
        default=0.01,
        help="Multiplicative factor of learning rate decay",
    )

    scheduler_config.add_argument(
        "--multi_step_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
    )

    scheduler_config.add_argument(
        "--task_scheduler",
        type=str,
        default="round_robin",
        choices=["sequential", "round_robin", "mixed"],
        help="task scheduler",
    )

    scheduler_config.add_argument(
        "--sequential_scheduler_fillup",
        type=str2bool,
        default=False,
        help="whether fillup in sequential scheduler",
    )

    scheduler_config.add_argument(
        "--round_robin_scheduler_fillup",
        type=str2bool,
        default=False,
        help="whether fillup in round robin scheduler",
    )

    scheduler_config.add_argument(
        "--mixed_scheduler_fillup",
        type=str2bool,
        default=False,
        help="whether fillup in mixed scheduler scheduler",
    )

    # Logging configuration
    logging_config = parser.add_argument_group("Logging configuration")

    logging_config.add_argument(
        "--counter_unit",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="Logging unit (epoch, batch)",
    )

    logging_config.add_argument(
        "--evaluation_freq", type=float, default=1, help="Logging evaluation frequency"
    )

    logging_config.add_argument(
        "--writer",
        type=str,
        default="tensorboard",
        choices=["json", "tensorboard"],
        help="The writer format (json, tensorboard)",
    )

    logging_config.add_argument(
        "--checkpointing",
        type=str2bool,
        default=False,
        help="Whether to checkpoint the model",
    )

    logging_config.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpointing path"
    )

    logging_config.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1,
        help="Checkpointing every k logging time",
    )

    logging_config.add_argument(
        "--checkpoint_metric",
        type=str2dict,
        default={"model/train/all/loss": "min"},
        help=(
            "Checkpointing metric (metric_name:mode), "
            "e.g., `model/train/all/loss:min`"
        ),
    )

    logging_config.add_argument(
        "--checkpoint_task_metrics",
        type=str2dict,
        default=None,
        help=(
            "Task specific checkpointing metric "
            "(metric_name1:mode1,metric_name2:mode2)"
        ),
    )

    logging_config.add_argument(
        "--checkpoint_runway",
        type=float,
        default=0,
        help="Checkpointing runway (no checkpointing before k checkpointing unit)",
    )

    logging_config.add_argument(
        "--clear_intermediate_checkpoints",
        type=str2bool,
        default=True,
        help="Whether to clear intermediate checkpoints",
    )

    logging_config.add_argument(
        "--clear_all_checkpoints",
        type=str2bool,
        default=False,
        help="Whether to clear all checkpoints",
    )

    return parser


def parse_arg_to_config(args: Namespace) -> Dict[str, Any]:
    r"""Parse the arguments to config dict

    Args:
      args(Namespace): The parsed namespace from argument parser.

    Returns:
      dict: The config dict.

    """
    config = {
        "meta_config": {
            "seed": args.seed,
            "verbose": args.verbose,
            "log_path": args.log_path,
        },
        "data_config": {
            "min_data_len": args.min_data_len,
            "max_data_len": args.max_data_len,
        },
        "model_config": {
            "model_path": args.model_path,
            "device": args.device,
            "dataparallel": args.dataparallel,
        },
        "learner_config": {
            "fp16": args.fp16,
            "n_epochs": args.n_epochs,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "test_split": args.test_split,
            "ignore_index": args.ignore_index,
            "optimizer_config": {
                "optimizer": args.optimizer,
                "lr": args.lr,
                "l2": args.l2,
                "grad_clip": args.grad_clip,
                "sgd_config": {
                    "momentum": args.sgd_momentum,
                    "dampening": args.sgd_dampening,
                    "nesterov": args.sgd_nesterov,
                },
                "adam_config": {
                    "betas": (0.9, 0.999),
                    "amsgrad": args.amsgrad,
                    "eps": args.eps,
                },
                "adamax_config": {"betas": (0.9, 0.999), "eps": args.eps},
                "bert_adam_config": {"betas": (0.9, 0.999), "eps": args.eps},
            },
            "lr_scheduler_config": {
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": args.warmup_steps,
                "warmup_unit": args.warmup_unit,
                "warmup_percentage": args.warmup_percentage,
                "min_lr": args.min_lr,
                "linear_config": {"min_lr": args.linear_lr_scheduler_min_lr},
                "exponential_config": {"gamma": args.exponential_lr_scheduler_gamma},
                "plateau_config": {
                    "factor": args.plateau_lr_scheduler_factor,
                    "patience": args.plateau_lr_scheduler_patience,
                    "threshold": args.plateau_lr_scheduler_threshold,
                },
                "step_config": {
                    "step_size": args.step_lr_scheduler_step_size,
                    "gamma": args.step_lr_scheduler_gamma,
                    "last_epoch": args.step_lr_scheduler_last_epoch,
                },
                "multi_step_config": {
                    "milestones": args.multi_step_lr_scheduler_milestones,
                    "gamma": args.multi_step_lr_scheduler_gamma,
                    "last_epoch": args.multi_step_lr_scheduler_last_epoch,
                },
            },
            "task_scheduler_config": {
                "task_scheduler": args.task_scheduler,
                "sequential_scheduler_config": {
                    "fillup": args.sequential_scheduler_fillup
                },
                "round_robin_scheduler_config": {
                    "fillup": args.round_robin_scheduler_fillup
                },
                "mixed_scheduler_config": {"fillup": args.mixed_scheduler_fillup},
            },
        },
        "logging_config": {
            "counter_unit": args.counter_unit,
            "evaluation_freq": args.evaluation_freq,
            "writer_config": {"writer": args.writer, "verbose": True},
            "checkpointing": args.checkpointing,
            "checkpointer_config": {
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_task_metrics": args.checkpoint_task_metrics,
                "checkpoint_runway": args.checkpoint_runway,
                "clear_intermediate_checkpoints": args.clear_intermediate_checkpoints,
                "clear_all_checkpoints": args.clear_all_checkpoints,
            },
        },
    }

    return config
