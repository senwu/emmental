import logging

from emmental.utils.parse_args import parse_args, parse_args_to_config

logger = logging.getLogger(__name__)


def test_parse_args(caplog):
    """Unit test of parsing args"""

    caplog.set_level(logging.INFO)

    parser = parse_args()
    args = parser.parse_args(["--seed", "0"])
    assert args.seed == 0

    config = parse_args_to_config(args)

    assert config == {
        "meta_config": {"seed": 0, "verbose": True, "log_path": "logs"},
        "data_config": {"min_data_len": 0, "max_data_len": 0},
        "model_config": {"model_path": None, "device": 0, "dataparallel": True},
        "learner_config": {
            "fp16": False,
            "n_epochs": 1,
            "train_split": ["train"],
            "valid_split": ["valid"],
            "test_split": ["test"],
            "ignore_index": None,
            "optimizer_config": {
                "optimizer": "adam",
                "lr": 0.001,
                "l2": 0.0,
                "grad_clip": None,
                "asgd_config": {"lambd": 0.0001, "alpha": 0.75, "t0": 1000000.0},
                "adadelta_config": {"rho": 0.9, "eps": 1e-06},
                "adagrad_config": {
                    "lr_decay": 0,
                    "initial_accumulator_value": 0,
                    "eps": 1e-10,
                },
                "adam_config": {"betas": (0.9, 0.999), "amsgrad": False, "eps": 1e-08},
                "adamw_config": {"betas": (0.9, 0.999), "amsgrad": False, "eps": 1e-08},
                "adamax_config": {"betas": (0.9, 0.999), "eps": 1e-08},
                "lbfgs_config": {
                    "max_iter": 20,
                    "max_eval": None,
                    "tolerance_grad": 1e-07,
                    "tolerance_change": 1e-09,
                    "history_size": 100,
                    "line_search_fn": None,
                },
                "rms_prop_config": {
                    "alpha": 0.99,
                    "eps": 1e-08,
                    "momentum": 0,
                    "centered": False,
                },
                "r_prop_config": {"etas": (0.5, 1.2), "step_sizes": (1e-06, 50)},
                "sgd_config": {"momentum": 0.0, "dampening": 0, "nesterov": False},
                "sparse_adam_config": {"betas": (0.9, 0.999), "eps": 1e-08},
                "bert_adam_config": {"betas": (0.9, 0.999), "eps": 1e-08},
            },
            "lr_scheduler_config": {
                "lr_scheduler": None,
                "warmup_steps": None,
                "warmup_unit": "batch",
                "warmup_percentage": None,
                "min_lr": 0.0,
                "linear_config": {"min_lr": 0.0},
                "exponential_config": {"gamma": 0.9},
                "plateau_config": {"factor": 0.5, "patience": 10, "threshold": 0.0001},
                "step_config": {"step_size": 1, "gamma": 0.01, "last_epoch": -1},
                "multi_step_config": {
                    "milestones": [10000],
                    "gamma": 0.01,
                    "last_epoch": -1,
                },
                "cosine_annealing_config": {"last_epoch": -1},
            },
            "task_scheduler_config": {
                "task_scheduler": "round_robin",
                "sequential_scheduler_config": {"fillup": False},
                "round_robin_scheduler_config": {"fillup": False},
                "mixed_scheduler_config": {"fillup": False},
            },
        },
        "logging_config": {
            "counter_unit": "epoch",
            "evaluation_freq": 1,
            "writer_config": {"writer": "tensorboard", "verbose": True},
            "checkpointing": False,
            "checkpointer_config": {
                "checkpoint_path": None,
                "checkpoint_freq": 1,
                "checkpoint_metric": {"model/train/all/loss": "min"},
                "checkpoint_task_metrics": None,
                "checkpoint_runway": 0,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": False,
            },
        },
    }
