"""Emmental round robin scheduler unit tests."""
import logging

import numpy as np
import torch

import emmental
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.schedulers.round_robin_scheduler import RoundRobinScheduler
from emmental.utils.utils import set_random_seed

logger = logging.getLogger(__name__)


def test_round_robin_scheduler(caplog):
    """Unit test of round robin scheduler."""
    caplog.set_level(logging.INFO)

    emmental.Meta.init()

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

    batch_task_names = [
        batch_data[-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names == [task2, task1, task2, task2, task1]

    scheduler = RoundRobinScheduler(fillup=True)

    assert scheduler.get_num_batches(dataloaders) == 6

    batch_task_names = [
        batch_data[-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names == [task2, task1, task2, task2, task1, task1]
