"""Emmental e2e with skipping learned."""
import logging
import shutil
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from emmental import (
    EmmentalDataLoader,
    EmmentalDataset,
    EmmentalLearner,
    EmmentalModel,
    EmmentalTask,
    Meta,
    Scorer,
    init,
)

logger = logging.getLogger(__name__)


def test_e2e_skip_trained(caplog):
    """Run an end-to-end test."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_e2e_skip_trained"
    use_exact_log_path = False
    Meta.reset()
    init(dirpath, use_exact_log_path=use_exact_log_path)

    # Generate synthetic data
    N = 500
    X = np.random.random((N, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int)

    X = [torch.Tensor(X[i]) for i in range(N)]
    # Create dataset and dataloader

    X_train, X_dev, X_test = (
        X[: int(0.8 * N)],
        X[int(0.8 * N) : int(0.9 * N)],
        X[int(0.9 * N) :],
    )
    Y_train, Y_dev, Y_test = (
        torch.tensor(Y[: int(0.8 * N)]),
        torch.tensor(Y[int(0.8 * N) : int(0.9 * N)]),
        torch.tensor(Y[int(0.9 * N) :]),
    )

    train_dataset = EmmentalDataset(
        name="synthetic",
        X_dict={"data": X_train},
        Y_dict={"label1": Y_train},
    )

    dev_dataset = EmmentalDataset(
        name="synthetic",
        X_dict={"data": X_dev},
        Y_dict={"label1": Y_dev},
    )

    test_dataset = EmmentalDataset(
        name="synthetic",
        X_dict={"data": X_test},
        Y_dict={"label1": Y_test},
    )

    task_to_label_dict = {"task1": "label1"}

    train_dataloader = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=train_dataset,
        split="train",
        batch_size=10,
    )
    dev_dataloader = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=dev_dataset,
        split="valid",
        batch_size=10,
    )
    test_dataloader = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=test_dataset,
        split="test",
        batch_size=10,
    )

    # Create task
    def ce_loss(task_name, immediate_output_dict, Y, active):
        module_name = f"{task_name}_pred_head"
        return F.cross_entropy(
            immediate_output_dict[module_name][0][active], (Y.view(-1))[active]
        )

    def output(task_name, immediate_output_dict):
        module_name = f"{task_name}_pred_head"
        return F.softmax(immediate_output_dict[module_name][0], dim=1)

    task_metrics = {"task1": ["accuracy"]}

    class IdentityModule(nn.Module):
        def __init__(self):
            """Initialize IdentityModule."""
            super().__init__()

        def forward(self, input):
            return {"out": input}

    tasks = [
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "input_module0": IdentityModule(),
                    "input_module1": nn.Linear(2, 8),
                    f"{task_name}_pred_head": nn.Linear(8, 2),
                }
            ),
            task_flow=[
                {
                    "name": "input",
                    "module": "input_module0",
                    "inputs": [("_input_", "data")],
                },
                {
                    "name": "input1",
                    "module": "input_module1",
                    "inputs": [("input", "out")],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": [("input1", 0)],
                },
            ],
            module_device={"input_module0": -1},
            loss_func=partial(ce_loss, task_name),
            output_func=partial(output, task_name),
            action_outputs=None,
            scorer=Scorer(metrics=task_metrics[task_name]),
            require_prob_for_eval=False,
            require_pred_for_eval=True,
        )
        for task_name in ["task1"]
    ]
    # Build model

    mtl_model = EmmentalModel(name="all", tasks=tasks)

    # Create learner
    emmental_learner = EmmentalLearner()

    config = {
        "meta_config": {"seed": 0, "verbose": False},
        "learner_config": {
            "n_steps": 200,
            "epochs_learned": 0,
            "steps_learned": 130,
            "skip_learned_data": True,
            "online_eval": True,
            "optimizer_config": {"lr": 0.01, "grad_clip": 100},
        },
        "logging_config": {
            "counter_unit": "epoch",
            "evaluation_freq": 0.2,
            "writer_config": {"writer": "json", "verbose": True},
            "checkpointing": True,
            "checkpointer_config": {
                "checkpoint_path": None,
                "checkpoint_freq": 1,
                "checkpoint_metric": {"model/all/train/loss": "min"},
                "checkpoint_task_metrics": None,
                "checkpoint_runway": 1,
                "checkpoint_all": False,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": True,
            },
        },
    }
    Meta.update_config(config)

    # Learning
    emmental_learner.learn(
        mtl_model,
        [train_dataloader, dev_dataloader],
    )

    test_score = mtl_model.score(test_dataloader)

    assert test_score["task1/synthetic/test/loss"] > 0.1

    Meta.reset()
    init(dirpath, use_exact_log_path=use_exact_log_path)

    config = {
        "meta_config": {"seed": 0, "verbose": False},
        "learner_config": {
            "n_steps": 200,
            "epochs_learned": 0,
            "steps_learned": 0,
            "skip_learned_data": False,
            "online_eval": True,
            "optimizer_config": {"lr": 0.01, "grad_clip": 100},
        },
        "logging_config": {
            "counter_unit": "epoch",
            "evaluation_freq": 0.2,
            "writer_config": {"writer": "tensorboard", "verbose": True},
            "checkpointing": True,
            "checkpointer_config": {
                "checkpoint_path": None,
                "checkpoint_freq": 1,
                "checkpoint_metric": {"model/all/train/loss": "min"},
                "checkpoint_task_metrics": None,
                "checkpoint_runway": 1,
                "checkpoint_all": False,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": True,
            },
        },
    }
    Meta.update_config(config)

    # Learning
    emmental_learner.learn(
        mtl_model,
        [train_dataloader, dev_dataloader],
    )

    test_score = mtl_model.score(test_dataloader)

    assert test_score["task1/synthetic/test/loss"] <= 0.1

    shutil.rmtree(dirpath)
