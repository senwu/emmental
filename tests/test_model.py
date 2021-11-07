"""Emmental model unit tests."""
import logging
import shutil
from functools import partial

import pytest
from torch import nn as nn
from torch.nn import functional as F

from emmental import EmmentalModel, EmmentalTask, Meta, Scorer, init
from emmental.modules.identity_module import IdentityModule

logger = logging.getLogger(__name__)


def test_model(caplog):
    """Unit test of model."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_model"

    Meta.reset()
    init(dirpath)

    def ce_loss(module_name, immediate_output_dict, Y):
        return F.cross_entropy(immediate_output_dict[module_name][0], Y.view(-1))

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
    Meta.update_config(config)

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

    # Save to a non-exit folder
    model.save(f"{dirpath}/new_create/saved_model.pth")

    # Save from a non-exit folder
    with pytest.raises(FileNotFoundError):
        model.load(f"{dirpath}/new_create_/saved_model.pth")

    # Test add_tasks
    model = EmmentalModel(name="test")

    model.add_tasks([task1, task2])
    assert model.task_names == set(["task_1", "task_2"])

    shutil.rmtree(dirpath)


def test_model_invalid_task(caplog):
    """Unit test of model with invalid task."""
    caplog.set_level(logging.INFO)

    dirpath = "temp_test_model_with_invalid_task"

    Meta.reset()
    init(
        dirpath,
        config={
            "meta_config": {"verbose": 0},
        },
    )

    task_name = "task1"

    task = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {
                "input_module0": IdentityModule(),
                f"{task_name}_pred_head": IdentityModule(),
            }
        ),
        task_flow=[
            {
                "name": "input1",
                "module": "input_module0",
                "inputs": [("_input_", "data")],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("input1", 0)],
            },
        ],
        module_device={"input_module0": -1},
        loss_func=None,
        output_func=None,
        action_outputs=None,
        scorer=None,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
    )

    task1 = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {
                "input_module0": IdentityModule(),
                f"{task_name}_pred_head": IdentityModule(),
            }
        ),
        task_flow=[
            {
                "name": "input1",
                "module": "input_module0",
                "inputs": [("_input_", "data")],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("input1", 0)],
            },
        ],
        module_device={"input_module0": -1},
        loss_func=None,
        output_func=None,
        action_outputs=None,
        scorer=None,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
    )

    model = EmmentalModel(name="test")
    model.add_task(task)

    model.remove_task(task_name)
    assert model.task_names == set([])

    model.remove_task("task_2")
    assert model.task_names == set([])

    model.add_task(task)

    # Duplicate task
    with pytest.raises(ValueError):
        model.add_task(task1)

    # Invalid task
    with pytest.raises(ValueError):
        model.add_task(task_name)

    shutil.rmtree(dirpath)


def test_get_data_from_output_dict(caplog):
    """Unit test of model get data from output_dict."""
    caplog.set_level(logging.INFO)

    output_dict = {
        "_input_": {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]},
        "feature_output": [8, 9, 10, 11],
        "pred_head": [[1, 0], [0, 1]],
        "feature3": 1,
        "_input1_": {"feature3": [1, 2, 3, 4], "feature4": [5, 6, 7, 8]},
    }

    model = EmmentalModel(name="test")

    input = "_input_"
    assert model._get_data_from_output_dict(output_dict, input) == {
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
    }

    input = "_input1_"
    assert model._get_data_from_output_dict(output_dict, input) == {
        "feature3": [1, 2, 3, 4],
        "feature4": [5, 6, 7, 8],
    }

    input = ("_input_", "feature1")
    assert model._get_data_from_output_dict(output_dict, input) == [1, 2, 3, 4]

    input = "feature_output"
    assert model._get_data_from_output_dict(output_dict, input) == [8, 9, 10, 11]

    input = ("pred_head", 0)
    assert model._get_data_from_output_dict(output_dict, input) == [1, 0]

    input = ("pred_head", 1)
    assert model._get_data_from_output_dict(output_dict, input) == [0, 1]

    with pytest.raises(ValueError):
        input = 1
        model._get_data_from_output_dict(output_dict, input)

    with pytest.raises(ValueError):
        input = ("_input_", 1)
        model._get_data_from_output_dict(output_dict, input)

    with pytest.raises(ValueError):
        input = ("feature_output", "a")
        model._get_data_from_output_dict(output_dict, input)

    with pytest.raises(ValueError):
        input = ("feature3", 1)
        model._get_data_from_output_dict(output_dict, input)
