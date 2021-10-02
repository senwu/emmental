"""Emmental e2e with no y dict test."""
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


def test_e2e_no_y_dict(caplog):
    """Run an end-to-end test."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_e2e_no_y_dict"
    use_exact_log_path = False
    Meta.reset()
    init(dirpath, use_exact_log_path=use_exact_log_path)

    config = {
        "meta_config": {"seed": 0, "verbose": False},
        "learner_config": {
            "n_epochs": 5,
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
        X_dict={"data": X_train, "label1": Y_train},
        Y_dict={"label2": Y_train},
    )

    dev_dataset = EmmentalDataset(
        name="synthetic",
        X_dict={"data": X_dev, "label1": Y_dev},
        Y_dict={"label2": Y_dev},
    )

    test_dataset = EmmentalDataset(
        name="synthetic",
        X_dict={"data": X_test, "label1": Y_test},
        Y_dict={"label2": Y_test},
    )

    task_to_label_dict = {"task1": None, "task2": "label2"}

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
    def ce_loss2(task_name, immediate_output_dict, Y):
        module_name = f"{task_name}_pred_head"
        return F.cross_entropy(immediate_output_dict[module_name][0], Y.view(-1))

    def ce_loss1(task_name, immediate_output_dict, Y):
        module_name = f"{task_name}_pred_head"
        return F.cross_entropy(
            immediate_output_dict[module_name][0],
            immediate_output_dict["_input_"]["label1"].view(-1),
        )

    def output(task_name, immediate_output_dict):
        module_name = f"{task_name}_pred_head"
        return F.softmax(immediate_output_dict[module_name][0], dim=1)

    task_metrics = {"task1": ["accuracy"], "task2": ["accuracy"]}

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
            loss_func=partial(
                ce_loss1 if task_name == "task1" else ce_loss2, task_name
            ),
            output_func=partial(output, task_name),
            action_outputs=[
                (f"{task_name}_pred_head", 0),
                ("_input_", "data"),
                (f"{task_name}_pred_head", 0),
            ]
            if task_name == "task2"
            else None,
            scorer=Scorer(metrics=task_metrics[task_name]),
            require_prob_for_eval=True if task_name in ["task2"] else False,
            require_pred_for_eval=True if task_name in ["task1"] else False,
        )
        for task_name in ["task1", "task2"]
    ]
    # Build model

    mtl_model = EmmentalModel(name="all", tasks=tasks)

    # Create learner
    emmental_learner = EmmentalLearner()

    # Learning
    emmental_learner.learn(
        mtl_model,
        [train_dataloader, dev_dataloader],
    )

    test_score = mtl_model.score(test_dataloader)

    assert test_score["task1/synthetic/test/loss"] <= 0.1
    assert test_score["task2/synthetic/test/loss"] <= 0.1
    assert test_score["task2/synthetic/test/accuracy"] >= 0.7

    shutil.rmtree(dirpath)
