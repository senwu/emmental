import logging
from functools import partial

from torch import nn

from emmental.contrib.slicing.modules import utils
from emmental.contrib.slicing.modules.slice_attention_module import SliceAttentionModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask

logger = logging.getLogger(__name__)


def build_slice_tasks(task, slice_func_dict):
    """A function to build slice tasks based on slicing functions.

    We assume the original task flow contains feature extractor and predictor head.
    - The name for feature extractor module should be {task.name}_feature
    - The name for predictor head should be {task.name}_pred_head

    For each slicing this function will create two corresponding tasks
    - A slice indicator task to learn whether the data sample is in the slice or not.
    - A slice predictor task that is only learned on the data samples in that slice

    All slice tasks are based on feature extractor module and a slice attention module
    will combine all slice task head to make the final predictions.
    """

    # Sanity check the task
    assert f"{task.name}_feature" in [
        action["name"] for action in task.task_flow
    ], f"{task.name}_feature should be in the task module_pool"

    assert f"{task.name}_pred_head" in [
        action["name"] for action in task.task_flow
    ], f"{task.name}_feature should be in the task module_pool"

    # Collect task predictor module info
    for action in task.task_flow:
        if f"{task.name}_pred_head" == action["name"]:
            base_task_predictor_action = action
            base_task_predictor_name = action["module"]
            base_task_predictor_module = task.module_pool[action["module"]]
            if isinstance(base_task_predictor_module, nn.DataParallel):
                base_task_predictor_module = base_task_predictor_module.module
            break

    task_feature_size = base_task_predictor_module.in_features
    task_cardinality = base_task_predictor_module.in_features

    # Remove the predictor head
    base_task_module_pool = task.module_pool
    del base_task_module_pool[base_task_predictor_name]

    base_task_task_flow = task.task_flow
    for idx, i in enumerate(base_task_task_flow):
        if i["name"] == f"{task.name}_pred_head":
            action_idx = idx
            break
    del base_task_task_flow[action_idx]

    tasks = []

    # Create slice indicator tasks.
    # (Note: indicator only has two classes, e.g, in the slice or out)
    for slice_name in slice_func_dict.keys():
        # Create task name
        ind_task_name = f"{task.name}_slice:ind_{slice_name}"

        # Create ind module
        ind_head_module_name = f"{ind_task_name}_head"
        ind_head_module = nn.Linear(task_feature_size, 2)

        # Create module_pool
        ind_module_pool = base_task_module_pool
        ind_module_pool[ind_head_module_name] = ind_head_module

        # Create task_flow
        ind_task_flow = base_task_task_flow
        ind_task_flow.extend(
            [
                {
                    "name": ind_head_module_name,
                    "module": ind_head_module_name,
                    "inputs": base_task_predictor_action["inputs"],
                }
            ]
        )

        tasks.append(
            EmmentalTask(
                name=ind_task_name,
                module_pool=ind_module_pool,
                task_flow=ind_task_flow,
                loss_func=partial(utils.ce_loss, ind_head_module_name),
                output_func=partial(utils.output, ind_head_module_name),
                scorer=Scorer(metrics=["f1"]),
            )
        )

    # Create slice predictor tasks

    # Create share predictor for all slice predictor
    shared_pred_head_module_name = f"{task.name}_slice:shared_pred"
    shared_pred_head_module = nn.Linear(task_feature_size, task_cardinality)

    for slice_name in slice_func_dict.keys():
        # Create task name
        pred_task_name = f"{task.name}_slice:pred_{slice_name}"

        # Create pred module
        pred_head_module_name = f"{pred_task_name}_head"
        pred_transform_module_name = f"{task.name}_slice:transform_{slice_name}"
        pred_transform_module = nn.Linear(task_feature_size, task_feature_size)

        pred_module_pool = base_task_module_pool
        pred_module_pool[pred_transform_module_name] = pred_transform_module
        pred_module_pool[shared_pred_head_module_name] = shared_pred_head_module

        # Create task_flow
        pred_task_flow = base_task_task_flow
        pred_task_flow.extend(
            [
                {
                    "name": pred_transform_module_name,
                    "module": pred_transform_module_name,
                    "inputs": base_task_predictor_action["inputs"],
                },
                {
                    "name": pred_head_module_name,
                    "module": shared_pred_head_module_name,
                    "inputs": [(pred_transform_module_name, 0)],
                },
            ]
        )

        tasks.append(
            EmmentalTask(
                name=pred_task_name,
                module_pool=pred_module_pool,
                task_flow=pred_task_flow,
                loss_func=partial(utils.ce_loss, pred_head_module_name),
                output_func=partial(utils.output, pred_head_module_name),
                scorer=task.scorer,
            )
        )

    # Create master task

    # Create task name
    master_task_name = task.name

    # Create attention module
    master_attention_module_name = f"{master_task_name}_attention"
    master_attention_module = SliceAttentionModule(
        slice_ind_key="_slice:ind_",
        slice_pred_key="_slice:pred_",
        slice_pred_feat_key="_slice:transform_",
    )

    # Create module pool
    master_head_module_name = f"{master_task_name}_head"
    master_head_module = base_task_predictor_module

    master_module_pool = nn.ModuleDict(
        {
            master_attention_module_name: master_attention_module,
            master_head_module_name: master_head_module,
        }
    )

    # Create task_flow
    master_task_flow = (
        ind_task_flow
        + pred_task_flow
        + [
            {
                "name": master_attention_module_name,
                "module": master_attention_module_name,
                "inputs": [],
            },
            {
                "name": master_head_module_name,
                "module": master_head_module_name,
                "inputs": [(master_attention_module_name, 0)],
            },
        ]
    )

    tasks.append(
        EmmentalTask(
            name=master_task_name,
            module_pool=master_module_pool,
            task_flow=master_task_flow,
            loss_func=partial(utils.ce_loss, master_head_module_name),
            output_func=partial(utils.output, master_head_module_name),
            scorer=task.scorer,
        )
    )

    return tasks
