"""Emmental sequential scheduler unit tests."""
import logging

import numpy as np
import torch

from emmental import EmmentalDataLoader, EmmentalDataset, init
from emmental.schedulers.sequential_scheduler import SequentialScheduler

logger = logging.getLogger(__name__)


def test_sequential_scheduler(caplog):
    """Unit test of sequential scheduler."""
    caplog.set_level(logging.INFO)

    init()

    task1 = "task1"
    x1 = np.random.rand(20, 2)
    y1 = torch.from_numpy(np.random.rand(20))

    task2 = "task2"
    x2 = np.random.rand(30, 3)
    y2 = torch.from_numpy(np.random.rand(30))

    dataloaders = [
        EmmentalDataLoader(
            task_to_label_dict={task_name: "label"},
            dataset=EmmentalDataset(
                name=task_name, X_dict={"feature": x}, Y_dict={"label": y}
            ),
            split="train",
            batch_size=10,
            shuffle=True,
        )
        for task_name, x, y in [(task1, x1, y1), (task2, x2, y2)]
    ]

    scheduler = SequentialScheduler()

    assert scheduler.get_num_batches(dataloaders) == 5

    batch_task_names = [
        batch_data[-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names == [task1, task1, task2, task2, task2]

    scheduler = SequentialScheduler(fillup=True)

    assert scheduler.get_num_batches(dataloaders) == 6

    batch_task_names = [
        batch_data[-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names == [task1, task1, task1, task2, task2, task2]


def test_sequential_scheduler_no_y_dict(caplog):
    """Unit test of sequential scheduler with no y_dict."""
    caplog.set_level(logging.INFO)

    init()

    task1 = "task1"
    x1 = np.random.rand(20, 2)

    task2 = "task2"
    x2 = np.random.rand(30, 3)

    dataloaders = [
        EmmentalDataLoader(
            task_to_label_dict={task_name: None},
            dataset=EmmentalDataset(name=task_name, X_dict={"feature": x}),
            split="train",
            batch_size=10,
            shuffle=True,
        )
        for task_name, x in [(task1, x1), (task2, x2)]
    ]

    dataloaders[0].n_batches = 3
    dataloaders[1].n_batches = 4

    scheduler = SequentialScheduler()

    assert scheduler.get_num_batches(dataloaders) == 7

    batch_task_names = [
        batch_data[2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names == [None] * 7
