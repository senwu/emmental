import logging
from typing import Callable, Dict, List, Tuple, Union

from torch import nn
from torch.nn.modules.container import ModuleDict

from emmental.meta import Meta
from emmental.scorer import Scorer

logger = logging.getLogger(__name__)


class EmmentalTask(object):
    r"""Task class to define task in Emmental model.

    Args:
      name(str): The name of the task (Primary key).
      module_pool(ModuleDict): A dict of modules that uses in the task.
      task_flow(list): The task flow among modules to define how the data flows.
      loss_func(callable): The function to calculate the loss.
      output_func(callable): The function to generate the output.
      scorer(Scorer): The class of metrics to evaluate the task.
      weight(float or int): The weight of the task.

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
        weight: Union[float, int] = 1.0,
    ) -> None:

        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer
        self.weight = weight

        if Meta.config["meta_config"]["verbose"]:
            logger.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
