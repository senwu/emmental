"""Emmental task."""
import logging
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.container import ModuleDict

from emmental.meta import Meta
from emmental.scorer import Scorer

logger = logging.getLogger(__name__)

ActionIndex = Union[str, Tuple[str, str], Tuple[str, int]]
ActionInputs = Union[str, Sequence[ActionIndex]]
ActionOutputs = Union[str, Sequence[ActionIndex]]


class EmmentalTaskFlowAction:
    """An action to execute in a EmmentalTask task_flow.

    Args:
      name: The name of the action.
      module_name: The module_name of the action.
      inputs: The inputs of the action. The inputs can be several format:
        1) X: takes all action X's output as input for this action.
        2) X, Y (int): takes action X's Yth output as input for this action.
        3) X, Y (str): takes action X's output Y as input for this action.
        4) None: takes all task_flow's output as input for this action.
    """

    def __init__(
        self,
        name: str,
        module: str,
        inputs: Optional[ActionInputs] = None,
    ) -> None:
        """Initialize Action."""
        self.name = name
        self.module = module
        self.inputs = inputs if inputs is None or isinstance(inputs, list) else [inputs]

    def __repr__(self) -> str:
        """Represent the action as a string."""
        return (
            f"Action(name={self.name}, "
            f"module={self.module}, "
            f"inputs={self.inputs})"
        )


class EmmentalTask(object):
    """Task class to define task in Emmental model.

    Args:
      name: The name of the task (Primary key).
      module_pool: A dict of modules that uses in the task.
      task_flow: The task flow among modules to define how the data flows.
      loss_func: The function to calculate the loss.
      output_func: The function to generate the output.
      scorer: The class of metrics to evaluate the task, defaults to None.
      action_outputs: The action outputs need to output, defaults to None.
      module_device: The dict of module device specification, defaults to None.
      weight: The weight of the task, defaults to 1.0.
      require_prob_for_eval: Whether require prob for evaluation, defaults to True.
      require_pred_for_eval: Whether require pred for evaluation, defaults to True.
    """

    def __init__(
        self,
        name: str,
        module_pool: ModuleDict,
        task_flow: Sequence[EmmentalTaskFlowAction],
        loss_func: Callable,
        output_func: Callable,
        scorer: Scorer = None,
        action_outputs: Optional[ActionOutputs] = None,
        module_device: Dict[str, Union[int, str, torch.device]] = dict(),
        weight: Union[float, int] = 1.0,
        require_prob_for_eval: bool = True,
        require_pred_for_eval: bool = True,
    ) -> None:
        """Initialize EmmentalTask."""
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer
        self.action_outputs = (
            action_outputs
            if action_outputs is None or isinstance(action_outputs, list)
            else [action_outputs]
        )
        if action_outputs is not None:
            self.action_outputs = list(set(action_outputs))
        self.module_device = {}
        for module_name in module_device.keys():
            if module_name not in self.module_pool:
                logger.warning(
                    f"Module {module_name} from module_device doesn't in module_pool, "
                    "skip..."
                )
                continue
            if module_device[module_name] == -1:
                self.module_device[module_name] = torch.device("cpu")
            else:
                self.module_device[module_name] = torch.device(
                    module_device[module_name]
                )
        self.require_prob_for_eval = require_prob_for_eval
        self.require_pred_for_eval = require_pred_for_eval
        self.weight = weight

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        """Represent the task as a string."""
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
