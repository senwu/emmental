import logging
from typing import Callable, Dict, List, Tuple, Union

from torch import nn
from torch.nn.modules.container import ModuleDict

from emmental.scorer import Scorer

logger = logging.getLogger(__name__)


class EmmentalTask(object):
    """Task class to define task in Emmental model.

    :param name: The name of the task (Primary key).
    :type name: str
    :param module_pool: A dict of modules that uses in the task.
    :type module_pool: nn.ModuleDict
    :param task_flow: The task flow among modules to define how the data flows.
    :type task_flow: list
    :param loss_func: The function to calculate the loss.
    :type loss_func: function
    :param output_func: The function to generate the output.
    :type output_func: function
    :param scorer: The class of metrics to evaluate the task.
    :type scorer: Scorer class
    :param weight: The weight of the task.
    :type scorer: float

    """

    def __init__(
        self,
        name: str,
        module_pool: ModuleDict,
        task_flow: List[
            Dict[str, Union[str, List[Tuple[str, str]], List[Tuple[str, int]]]]
        ],
        loss_func: Callable,
        output_func: Callable,
        scorer: Scorer,
        weight: float = 1.0,
    ) -> None:
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer
        self.weight = weight

        logger.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
