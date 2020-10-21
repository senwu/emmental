"""Emmental model unit tests."""
import logging
import shutil
from functools import partial

from torch import nn as nn
from torch.nn import functional as F

import emmental
from emmental import Meta
from emmental.model import EmmentalModel
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

logger = logging.getLogger(__name__)


def test_model(caplog):
    """Unit test of model."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_model"

    Meta.reset()
    emmental.init(dirpath)

    def ce_loss(module_name, immediate_output_dict, Y, active):
        return F.cross_entropy(
            immediate_output_dict[module_name][0][active], (Y.view(-1))[active]
        )

    def output(module_name, immediate_output_dict):
        return F.softmax(immediate_output_dict[module_name][0], dim=1)

    task1 = EmmentalTask(
        name="task_1",
        module_pool=nn.ModuleDict(
            {"m1": nn.Linear(10, 10, bias=False), "m2": nn.Linear(10, 2, bias=False)}
        ),
        task_flow=[
            {"name": "m1", "module": "m1", "inputs": [("_input_", "data")]},
            {"name": "m2", "module": "m2", "inputs": [("m1", 0)]},
        ],
        loss_func=partial(ce_loss, "m2"),
        output_func=partial(output, "m2"),
        scorer=Scorer(metrics=["accuracy"]),
    )

    new_task1 = EmmentalTask(
        name="task_1",
        module_pool=nn.ModuleDict(
            {"m1": nn.Linear(10, 5, bias=False), "m2": nn.Linear(5, 2, bias=False)}
        ),
        task_flow=[
            {"name": "m1", "module": "m1", "inputs": [("_input_", "data")]},
            {"name": "m2", "module": "m2", "inputs": [("m1", 0)]},
        ],
        loss_func=partial(ce_loss, "m2"),
        output_func=partial(output, "m2"),
        scorer=Scorer(metrics=["accuracy"]),
    )

    task2 = EmmentalTask(
        name="task_2",
        module_pool=nn.ModuleDict(
            {"m1": nn.Linear(10, 5, bias=False), "m2": nn.Linear(5, 2, bias=False)}
        ),
        task_flow=[
            {"name": "m1", "module": "m1", "inputs": [("_input_", "data")]},
            {"name": "m2", "module": "m2", "inputs": [("m1", 0)]},
        ],
        loss_func=partial(ce_loss, "m2"),
        output_func=partial(output, "m2"),
        scorer=Scorer(metrics=["accuracy"]),
    )

    config = {"model_config": {"dataparallel": False}}
    emmental.Meta.update_config(config)

    model = EmmentalModel(name="test", tasks=task1)

    assert repr(model) == "EmmentalModel(name=test)"
    assert model.name == "test"
    assert model.task_names == set(["task_1"])
    assert model.module_pool["m1"].weight.data.size() == (10, 10)
    assert model.module_pool["m2"].weight.data.size() == (2, 10)

    model.update_task(new_task1)

    assert model.module_pool["m1"].weight.data.size() == (5, 10)
    assert model.module_pool["m2"].weight.data.size() == (2, 5)

    model.update_task(task2)

    assert model.task_names == set(["task_1"])

    model.add_task(task2)

    assert model.task_names == set(["task_1", "task_2"])

    model.remove_task("task_1")
    assert model.task_names == set(["task_2"])

    model.remove_task("task_1")
    assert model.task_names == set(["task_2"])

    model.save(f"{dirpath}/saved_model.pth")

    model.load(f"{dirpath}/saved_model.pth")

    # Test add_tasks
    model = EmmentalModel(name="test")

    model.add_tasks([task1, task2])
    assert model.task_names == set(["task_1", "task_2"])

    shutil.rmtree(dirpath)
