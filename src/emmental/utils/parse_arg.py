import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2dict(v):
    dict = {}
    for token in v.split(";"):
        key, value = token.split(":")
        dict[key] = value

    return dict


def parse_arg(parser=None):
    """Parse the command line"""
    if parser is None:
        parser = argparse.ArgumentParser(
            "Emmental configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    # Load meta configuration
    meta_config = parser.add_argument_group("Meta configuration")

    meta_config.add_argument(
        "--seed",
        type=int,
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

    # Load model configuration
    model_config = parser.add_argument_group("Model configuration")

    model_config.add_argument(
        "--model_path", type=str, default=None, help="Path to pretrained model"
    )

    model_config.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which device to use (-1 for cpu or gpu id (e.g., 0 for cuda:0))",
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
        "--valid_split", type=str, default="valid", help="The split for validation"
    )

    learner_config.add_argument(
        "--test_split", type=str, default="test", help="The split for testing"
    )

    learner_config.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="The ignore index, uses for masking samples",
    )

    # Optimizer configuration
    optimizer_config = parser.add_argument_group("Optimizer configuration")

    optimizer_config.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="The optimizer to use",
    )

    optimizer_config.add_argument("--lr", type=float, default=1e-3, help="Learing rate")

    optimizer_config.add_argument(
        "--l2", type=float, default=0.0, help="l2 regularization"
    )

    optimizer_config.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )

    optimizer_config.add_argument(
        "--sgd_momentum", type=float, default=0.9, help="SGD momentum"
    )

    # TODO: add adam betas

    # Scheduler configuration
    scheduler_config = parser.add_argument_group("Scheduler configuration")

    scheduler_config.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        choices=["linear", "exponential", "reduce_on_plateau"],
        help="Learning rate scheduler",
    )

    scheduler_config.add_argument(
        "--warmup_steps", type=float, default=0.0, help="Warm up steps"
    )

    scheduler_config.add_argument(
        "--warmup_unit",
        type=str,
        default="batch",
        choices=["epoch", "batch"],
        help="Warm up unit",
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
        "--task_scheduler",
        type=str,
        default="round_robin",
        choices=["sequential", "round_robin"],
        help="task scheduler",
    )

    # Logging configuration
    logging_config = parser.add_argument_group("Logging configuration")

    logging_config.add_argument(
        "--counter_unit",
        type=str,
        default="batch",
        choices=["epoch", "batch"],
        help="Logging unit (epoch, batch)",
    )

    logging_config.add_argument(
        "--evaluation_freq", type=int, default=2, help="Logging evaluation frequency"
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
        default=True,
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
            "Checkpointing metric (metric_name:mode), ",
            "e.g., `model/train/all/loss:min`",
        ),
    )

    logging_config.add_argument(
        "--checkpoint_task_metrics",
        type=str2dict,
        default=None,
        help=(
            "Task specific checkpointing metric ",
            "(metric_name1:mode2;metric_name2:mode2)",
        ),
    )

    logging_config.add_argument(
        "--checkpoint_runway",
        type=int,
        default=0,
        help="Checkpointing runway (no checkpointing before k checkpointing unit)",
    )

    logging_config.add_argument(
        "--checkpoint_clear",
        type=str2bool,
        default=True,
        help="Whether to clear immedidate checkpointing",
    )

    return parser


def parse_arg_to_config(args):
    """Parse the arguments to config dict"""

    config = {
        "meta_config": {
            "seed": args.seed,
            "verbose": args.verbose,
            "log_path": args.log_path,
        },
        "model_config": {"model_path": args.model_path, "device": args.device},
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
                "sgd_config": {"momentum": args.sgd_momentum},
                "adam_config": {"betas": (0.9, 0.999)},
            },
            "lr_scheduler_config": {
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": args.warmup_steps,
                "warmup_unit": args.warmup_unit,
                "min_lr": args.min_lr,
                "linear_config": {"min_lr": args.linear_lr_scheduler_min_lr},
                "exponential_config": {"gamma": args.exponential_lr_scheduler_gamma},
                "plateau_config": {
                    "factor": args.plateau_lr_scheduler_factor,
                    "patience": args.plateau_lr_scheduler_patience,
                    "threshold": args.plateau_lr_scheduler_threshold,
                },
            },
            "task_scheduler": args.task_scheduler,
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
                "checkpoint_clear": args.checkpoint_clear,
            },
        },
    }

    return config
