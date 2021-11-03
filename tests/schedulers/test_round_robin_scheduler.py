"""Emmental round robin scheduler unit tests."""
import logging

import numpy as np
import torch

from emmental import EmmentalDataLoader, EmmentalDataset, init
from emmental.schedulers.round_robin_scheduler import RoundRobinScheduler
from emmental.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def test_round_robin_scheduler(caplog):
    """Unit test of round robin scheduler."""
    caplog.set_level(logging.INFO)

    init()

    # Set random seed seed
    set_random_seed(2)

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

    scheduler = RoundRobinScheduler()

    assert scheduler.get_num_batches(dataloaders) == 5

    batch_data_names = [batch.data_name for batch in scheduler.get_batches(dataloaders)]

    assert batch_data_names == [task2, task1, task2, task2, task1]

    scheduler = RoundRobinScheduler(fillup=True)

    assert scheduler.get_num_batches(dataloaders) == 6

    batch_data_names = [batch.data_name for batch in scheduler.get_batches(dataloaders)]

    assert batch_data_names == [task2, task1, task2, task2, task1, task1]


def test_round_robin_scheduler_no_y_dict(caplog):
    """Unit test of round robin scheduler with no y_dict."""
    caplog.set_level(logging.INFO)

    init()

    # Set random seed seed
    set_random_seed(2)

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

    scheduler = RoundRobinScheduler()

    assert scheduler.get_num_batches(dataloaders) == 7

    batch_y_dicts = [batch.Y_dict for batch in scheduler.get_batches(dataloaders)]

    assert batch_y_dicts == [None] * 7
