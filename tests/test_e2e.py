"""Emmental e2e test."""
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


def test_e2e(caplog):
    """Run an end-to-end test."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_e2e"
    use_exact_log_path = False
    Meta.reset()
    init(dirpath, use_exact_log_path=use_exact_log_path)

    config = {
        "meta_config": {"seed": 0},
        "learner_config": {
            "n_epochs": 3,
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
    Y1 = (X[:, 0] > X[:, 1] + 0.25).astype(int)
    Y2 = (X[:, 0] > X[:, 1] + 0.2).astype(int)

    X = [torch.Tensor(X[i]) for i in range(N)]
    # Create dataset and dataloader

    X_train, X_dev, X_test = (
        X[: int(0.8 * N)],
        X[int(0.8 * N) : int(0.9 * N)],
        X[int(0.9 * N) :],
    )
    Y1_train, Y1_dev, Y1_test = (
        torch.tensor(Y1[: int(0.8 * N)]),
        torch.tensor(Y1[int(0.8 * N) : int(0.9 * N)]),
        torch.tensor(Y1[int(0.9 * N) :]),
    )
    Y2_train, Y2_dev, Y2_test = (
        torch.tensor(Y2[: int(0.8 * N)]),
        torch.tensor(Y2[int(0.8 * N) : int(0.9 * N)]),
        torch.tensor(Y2[int(0.9 * N) :]),
    )

    train_dataset1 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_train}, Y_dict={"label1": Y1_train}
    )

    train_dataset2 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_train}, Y_dict={"label2": Y2_train}
    )

    dev_dataset1 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_dev}, Y_dict={"label1": Y1_dev}
    )

    dev_dataset2 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_dev}, Y_dict={"label2": Y2_dev}
    )

    test_dataset1 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_test}, Y_dict={"label1": Y1_test}
    )

    test_dataset2 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_test}, Y_dict={"label2": Y2_test}
    )

    test_dataset3 = EmmentalDataset(name="synthetic", X_dict={"data": X_test})

    task_to_label_dict = {"task1": "label1"}

    train_dataloader1 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=train_dataset1,
        split="train",
        batch_size=10,
    )
    dev_dataloader1 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=dev_dataset1,
        split="valid",
        batch_size=10,
    )
    test_dataloader1 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=test_dataset1,
        split="test",
        batch_size=10,
    )

    task_to_label_dict = {"task2": "label2"}

    train_dataloader2 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=train_dataset2,
        split="train",
        batch_size=10,
    )
    dev_dataloader2 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=dev_dataset2,
        split="valid",
        batch_size=10,
    )
    test_dataloader2 = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=test_dataset2,
        split="test",
        batch_size=10,
    )

    test_dataloader3 = EmmentalDataLoader(
        task_to_label_dict=["task2"], dataset=test_dataset3, split="test", batch_size=10
    )

    # Create task
    def ce_loss(task_name, immediate_ouput_dict, Y, active):
        module_name = f"{task_name}_pred_head"
        return F.cross_entropy(
            immediate_ouput_dict[module_name][0][active], (Y.view(-1))[active]
        )

    def output(task_name, immediate_ouput_dict):
        module_name = f"{task_name}_pred_head"
        return F.softmax(immediate_ouput_dict[module_name][0], dim=1)

    task_metrics = {"task1": ["accuracy"], "task2": ["accuracy", "roc_auc"]}

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
        [train_dataloader1, train_dataloader2, dev_dataloader1, dev_dataloader2],
    )

    test1_score = mtl_model.score(test_dataloader1)
    test2_score = mtl_model.score(test_dataloader2)

    assert test1_score["task1/synthetic/test/accuracy"] >= 0.7
    assert (
        test1_score["model/all/test/macro_average"]
        == test1_score["task1/synthetic/test/accuracy"]
    )
    assert test2_score["task2/synthetic/test/accuracy"] >= 0.7
    assert test2_score["task2/synthetic/test/roc_auc"] >= 0.7

    test3_score = mtl_model.score(test_dataloader3)
    assert test3_score == {}

    test2_pred = mtl_model.predict(test_dataloader2, return_action_outputs=True)
    test3_pred = mtl_model.predict(test_dataloader3, return_action_outputs=True)

    assert test2_pred["uids"] == test3_pred["uids"]
    assert False not in [
        np.array_equal(
            test2_pred["probs"]["task2"][idx], test3_pred["probs"]["task2"][idx]
        )
        for idx in range(len(test3_pred["probs"]["task2"]))
    ]
    assert "outputs" in test2_pred
    assert "outputs" in test3_pred
    assert False not in [
        np.array_equal(
            test2_pred["outputs"]["task2"]["task2_pred_head_0"][idx],
            test3_pred["outputs"]["task2"]["task2_pred_head_0"][idx],
        )
        for idx in range(len(test2_pred["outputs"]["task2"]["task2_pred_head_0"]))
    ]
    assert False not in [
        np.array_equal(
            test2_pred["outputs"]["task2"]["_input__data"][idx],
            test3_pred["outputs"]["task2"]["_input__data"][idx],
        )
        for idx in range(len(test2_pred["outputs"]["task2"]["_input__data"]))
    ]

    test4_pred = mtl_model.predict(test_dataloader2, return_action_outputs=False)
    assert "outputs" not in test4_pred

    shutil.rmtree(dirpath)
