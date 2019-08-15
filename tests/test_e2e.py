import logging
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

logger = logging.getLogger(__name__)


def test_e2e(caplog):
    """Run an end-to-end test."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_e2e"

    Meta.reset()
    emmental.init(dirpath)

    # Generate synthetic data
    N = 50
    X = np.random.random((N, 2)) * 2 - 1
    Y1 = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1
    Y2 = (-X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    # Create dataset and dataloader

    splits = [0.8, 0.1, 0.1]

    X_train, X_dev, X_test = [], [], []
    Y1_train, Y1_dev, Y1_test = [], [], []
    Y2_train, Y2_dev, Y2_test = [], [], []

    for i in range(N):
        if i <= N * splits[0]:
            X_train.append(torch.Tensor(X[i]))
            Y1_train.append(Y1[i])
            Y2_train.append(Y2[i])
        elif i < N * (splits[0] + splits[1]):
            X_dev.append(torch.Tensor(X[i]))
            Y1_dev.append(Y1[i])
            Y2_dev.append(Y2[i])
        else:
            X_test.append(torch.Tensor(X[i]))
            Y1_test.append(Y1[i])
            Y2_test.append(Y2[i])

    Y1_train = torch.from_numpy(np.array(Y1_train))
    Y1_dev = torch.from_numpy(np.array(Y1_dev))
    Y1_test = torch.from_numpy(np.array(Y1_test))

    Y2_train = torch.from_numpy(np.array(Y1_train))
    Y2_dev = torch.from_numpy(np.array(Y2_dev))
    Y2_test = torch.from_numpy(np.array(Y2_test))

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
        name="synthetic", X_dict={"data": X_test}, Y_dict={"label1": Y2_test}
    )

    test_dataset2 = EmmentalDataset(
        name="synthetic", X_dict={"data": X_test}, Y_dict={"label2": Y2_test}
    )

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

    # Create task
    def ce_loss(task_name, immediate_ouput_dict, Y, active):
        module_name = f"{task_name}_pred_head"
        return F.cross_entropy(
            immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
        )

    def output(task_name, immediate_ouput_dict):
        module_name = f"{task_name}_pred_head"
        return F.softmax(immediate_ouput_dict[module_name][0], dim=1)

    task_name = "task1"

    task1 = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {"input_module": nn.Linear(2, 8), f"{task_name}_pred_head": nn.Linear(8, 2)}
        ),
        task_flow=[
            {
                "name": "input",
                "module": "input_module",
                "inputs": [("_input_", "data")],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("input", 0)],
            },
        ],
        loss_func=partial(ce_loss, task_name),
        output_func=partial(output, task_name),
        scorer=Scorer(metrics=["accuracy"]),
    )

    task_name = "task2"

    task2 = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {"input_module": nn.Linear(2, 8), f"{task_name}_pred_head": nn.Linear(8, 2)}
        ),
        task_flow=[
            {
                "name": "input",
                "module": "input_module",
                "inputs": [("_input_", "data")],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("input", 0)],
            },
        ],
        loss_func=partial(ce_loss, task_name),
        output_func=partial(output, task_name),
        scorer=Scorer(metrics=["accuracy", "roc_auc"]),
    )

    # Build model

    mtl_model = EmmentalModel(name="all", tasks=[task1, task2])

    # Create learner

    emmental_learner = EmmentalLearner()

    # Update learning config
    Meta.update_config(
        config={"learner_config": {"n_epochs": 10, "optimizer_config": {"lr": 0.01}}}
    )

    # Learning
    emmental_learner.learn(
        mtl_model,
        [train_dataloader1, train_dataloader2, dev_dataloader1, dev_dataloader2],
    )

    test1_score = mtl_model.score(test_dataloader1)
    test2_score = mtl_model.score(test_dataloader2)

    assert test1_score["task1/synthetic/test/accuracy"] >= 0.5
    assert (
        test1_score["model/all/test/macro_average"]
        == test1_score["task1/synthetic/test/accuracy"]
    )
    assert test2_score["task2/synthetic/test/accuracy"] >= 0.5
    assert test2_score["task2/synthetic/test/roc_auc"] >= 0.6
    assert test2_score["model/all/test/macro_average"] >= 0.6

    shutil.rmtree(dirpath)
