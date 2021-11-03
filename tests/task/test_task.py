"""Emmental task unit tests."""
import logging
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

import emmental
from emmental import EmmentalTask, EmmentalTaskFlowAction as Action, Scorer
from emmental.modules.identity_module import IdentityModule


def test_emmental_task(caplog):
    """Unit test of emmental task."""
    caplog.set_level(logging.INFO)

    emmental.init()

    def ce_loss(module_name, output_dict, Y):
        return F.cross_entropy(output_dict[module_name][0], Y.view(-1))

    def output(module_name, output_dict):
        return F.softmax(output_dict[module_name][0], dim=1)

    task_name = "task1"
    task_metrics = {task_name: ["accuracy"]}
    scorer = Scorer(metrics=task_metrics[task_name])

    task = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {
                "input_module0": IdentityModule(),
                "input_module1": IdentityModule(),
                f"{task_name}_pred_head": IdentityModule(),
            }
        ),
        task_flow=[
            Action("input1", "input_module0", [("_input_", "data")]),
            Action("input2", "input_module1", [("input1", 0)]),
            Action(f"{task_name}_pred_head", f"{task_name}_pred_head", [("input2", 0)]),
        ],
        module_device={"input_module0": -1, "input_module1": 0, "input_module": -1},
        loss_func=partial(ce_loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        action_outputs=None,
        scorer=scorer,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        weight=2.0,
    )

    assert task.name == task_name
    assert set(list(task.module_pool.keys())) == set(
        ["input_module0", "input_module1", f"{task_name}_pred_head"]
    )
    assert task.action_outputs is None
    assert task.scorer == scorer
    assert len(task.task_flow) == 3
    assert task.module_device == {
        "input_module0": torch.device("cpu"),
        "input_module1": torch.device(0),
    }
    assert task.require_prob_for_eval is False
    assert task.require_pred_for_eval is True
    assert task.weight == 2.0
