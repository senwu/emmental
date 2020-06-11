"""Emmental mixed scheduler unit tests."""
import logging

import numpy as np
import torch

import emmental
from emmental.data import EmmentalDataLoader, EmmentalDataset
from emmental.schedulers.mixed_scheduler import MixedScheduler

logger = logging.getLogger(__name__)


def test_mixed_scheduler(caplog):
    """Unit test of mixed scheduler."""
    caplog.set_level(logging.INFO)

    emmental.Meta.init()

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

    scheduler = MixedScheduler()

    assert scheduler.get_num_batches(dataloaders) == 2

    batch_task_names_1 = [
        batch_data[0][-2] for batch_data in scheduler.get_batches(dataloaders)
    ]
    batch_task_names_2 = [
        batch_data[1][-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names_1 == [task1, task1]
    assert batch_task_names_2 == [task2, task2]

    scheduler = MixedScheduler(fillup=True)

    assert scheduler.get_num_batches(dataloaders) == 3

    batch_task_names_1 = [
        batch_data[0][-2] for batch_data in scheduler.get_batches(dataloaders)
    ]
    batch_task_names_2 = [
        batch_data[1][-2] for batch_data in scheduler.get_batches(dataloaders)
    ]

    assert batch_task_names_1 == [task1, task1, task1]
    assert batch_task_names_2 == [task2, task2, task2]
