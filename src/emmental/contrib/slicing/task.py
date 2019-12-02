import copy
import logging
from functools import partial
from typing import Callable, Dict, List, Optional

from torch import Tensor, nn

from emmental.contrib.slicing.modules import utils
from emmental.contrib.slicing.modules.slice_attention_module import SliceAttentionModule
from emmental.meta import Meta
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from emmental.utils.utils import move_to_device

logger = logging.getLogger(__name__)


def build_slice_tasks(
    task: EmmentalTask,
    slice_func_dict: Dict[str, Callable],
    slice_scorer: Optional[Scorer] = None,
    slice_distribution: Dict[str, Tensor] = {},
    dropout: float = 0.0,
    slice_ind_head_module: Optional[nn.Module] = None,
    sep_slice_ind_feature: bool = False,
) -> List[EmmentalTask]:
    r"""A function to build slice tasks based on slicing functions.

      We assume the original task flow contains feature extractor and predictor head.
      - The predictor head action should be the last action
      - The feature extractor action should be input of the predictor head action

      For each slicing this function will create two corresponding tasks
      - A slice indicator task to learn whether the data sample is in the slice or not.
      - A slice predictor task that is only learned on the data samples in that slice

      All slice tasks are based on feature extractor module and a slice attention
      module will combine all slice task head to make the final predictions.

    Args:
      task(EmmentalTask): Task to do slicing learning.
      slice_func_dict(dict): Slicing functions.
      slice_scorer(Scorer): Slice scorer, defaults to None.
      slice_distribution(dict): Slice data class distribution, defaults to {}.
      dropout(float): Dropout, defaults to 0.0.
      slice_ind_head_module(nn.Module, optional): Slice indicator head module,
        defaults to None.
      sep_slice_ind_feature(bool): Whether to use sep slice ind feature,
        defaults to False.

    Returns:
      List[EmmentalTask]: list of tasks.

    """

    # Collect task predictor module info
    base_task_predictor_action = task.task_flow[-1]
    base_task_predictor_module = task.module_pool[
        base_task_predictor_action["module"]  # type: ignore
    ]
    if isinstance(base_task_predictor_module, nn.DataParallel):
        base_task_predictor_module = base_task_predictor_module.module

    task_feature_size = base_task_predictor_module.in_features
    task_cardinality = base_task_predictor_module.out_features

    # Remove the predictor head module and action
    base_task_module_pool = task.module_pool
    del base_task_module_pool[base_task_predictor_action["module"]]  # type: ignore

    base_task_task_flow = task.task_flow[:-1]

    tasks = []
    slice_module_pool = nn.ModuleDict()
    for module_name, module in task.module_pool.items():
        slice_module_pool[module_name] = module
    slice_actions = [action for action in base_task_task_flow]

    if slice_ind_head_module is None:
        slice_ind_head_module = nn.Linear(task_feature_size, 2)

    assert isinstance(slice_ind_head_module, nn.Module)

    if slice_scorer is None or not isinstance(slice_scorer, Scorer):
        slice_scorer = Scorer(metrics=["f1"])

    # Create slice indicator tasks.
    # (Note: indicator only has two classes, e.g, in the slice or out)
    for slice_name in slice_func_dict.keys():
        # Create task name
        ind_task_name = f"{task.name}_slice:ind_{slice_name}"

        # Create ind module
        ind_head_module_name = f"{ind_task_name}_head"
        ind_head_module = copy.deepcopy(slice_ind_head_module)

        ind_head_dropout_module_name = f"{task.name}_slice:dropout_{slice_name}"
        ind_head_dropout_module = nn.Dropout(p=dropout)

        # Create module_pool
        ind_module_pool = nn.ModuleDict(
            {
                module_name: module
                for module_name, module in base_task_module_pool.items()
            }
        )
        ind_module_pool[ind_head_dropout_module_name] = ind_head_dropout_module
        ind_module_pool[ind_head_module_name] = ind_head_module

        assert len(base_task_predictor_action["inputs"]) == 1

        ind_head_dropout_module_input_name = base_task_predictor_action["inputs"][0][0]
        ind_head_dropout_module_input_idx = 1 if sep_slice_ind_feature else 0

        # Create task_flow
        ind_task_flow = [action for action in base_task_task_flow]
        ind_task_flow.extend(
            [
                {
                    "name": ind_head_dropout_module_name,
                    "module": ind_head_dropout_module_name,
                    "inputs": [
                        (
                            ind_head_dropout_module_input_name,
                            ind_head_dropout_module_input_idx,
                        )
                    ],
                },
                {
                    "name": ind_head_module_name,
                    "module": ind_head_module_name,
                    "inputs": [(ind_head_dropout_module_name, 0)],
                },
            ]
        )

        # Add slice specific module to slice_module_pool
        slice_module_pool[ind_head_module_name] = ind_head_module
        slice_actions.extend(
            [
                {
                    "name": ind_head_dropout_module_name,
                    "module": ind_head_dropout_module_name,
                    "inputs": [
                        (
                            ind_head_dropout_module_input_name,
                            ind_head_dropout_module_input_idx,
                        )
                    ],
                },
                {
                    "name": ind_head_module_name,
                    "module": ind_head_module_name,
                    "inputs": [(ind_head_dropout_module_name, 0)],
                },
            ]
        )

        # Loss function
        if ind_task_name in slice_distribution:
            loss = partial(
                utils.ce_loss,
                ind_head_module_name,
                weight=move_to_device(
                    slice_distribution[ind_task_name],
                    Meta.config["model_config"]["device"],
                ),
            )
        else:
            loss = partial(utils.ce_loss, ind_head_module_name)

        tasks.append(
            EmmentalTask(
                name=ind_task_name,
                module_pool=ind_module_pool,
                task_flow=ind_task_flow,
                loss_func=loss,
                output_func=partial(utils.output, ind_head_module_name),
                scorer=slice_scorer,
            )
        )

    # Create slice predictor tasks

    # Create share predictor for all slice predictor
    shared_pred_head_module_name = f"{task.name}_slice:shared_pred"
    shared_pred_head_module = nn.Linear(task_feature_size, task_cardinality)

    # Add slice specific module to slice_module_pool
    slice_module_pool[shared_pred_head_module_name] = shared_pred_head_module

    for slice_name in slice_func_dict.keys():
        # Create task name
        pred_task_name = f"{task.name}_slice:pred_{slice_name}"

        # Create pred module
        pred_head_module_name = f"{pred_task_name}_head"
        pred_transform_module_name = f"{task.name}_slice:transform_{slice_name}"
        pred_transform_module = nn.Linear(task_feature_size, task_feature_size)

        # Create module_pool
        pred_module_pool = nn.ModuleDict(
            {
                module_name: module
                for module_name, module in base_task_module_pool.items()
            }
        )
        pred_module_pool[pred_transform_module_name] = pred_transform_module
        pred_module_pool[shared_pred_head_module_name] = shared_pred_head_module

        # Create task_flow
        pred_task_flow = [action for action in base_task_task_flow]
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

        # Add slice specific module to slice_module_pool
        slice_module_pool[pred_transform_module_name] = pred_transform_module
        slice_actions.extend(
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

        # Loss function
        if pred_task_name in slice_distribution:
            loss = partial(
                utils.ce_loss,
                pred_head_module_name,
                weight=move_to_device(
                    slice_distribution[pred_task_name],
                    Meta.config["model_config"]["device"],
                ),
            )
        else:
            loss = partial(utils.ce_loss, pred_head_module_name)

        tasks.append(
            EmmentalTask(
                name=pred_task_name,
                module_pool=pred_module_pool,
                task_flow=pred_task_flow,
                loss_func=loss,
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

    master_module_pool = slice_module_pool
    master_module_pool[master_attention_module_name] = master_attention_module
    master_module_pool[master_head_module_name] = master_head_module

    # Create task_flow
    master_task_flow = slice_actions + [
        {
            "name": master_attention_module_name,
            "module": master_attention_module_name,
            "inputs": [],  # type: ignore
        },
        {
            "name": master_head_module_name,
            "module": master_head_module_name,
            "inputs": [(master_attention_module_name, 0)],
        },
    ]

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
