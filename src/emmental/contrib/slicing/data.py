import logging
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch import Tensor

from emmental.contrib.slicing.slicing_function import slicing_function
from emmental.data import EmmentalDataLoader
from emmental.task import EmmentalTask

logger = logging.getLogger(__name__)


def add_slice_labels(
    task: EmmentalTask,
    dataloaders: List[EmmentalDataLoader],
    slice_func_dict: Dict[str, Callable],
    split: str = "train",
) -> Dict[str, Tensor]:
    r"""A function to extend dataloader by adding slice indicator and predictor
    labels.

    Args:
      task(EmmentalTask): Task to add slices.
      dataloaders(List[EmmentalDataLoader]): List of dataloaders to train on the task.
      slice_func_dict(dict): Slicing functions.
      split(str): Split to use, defaults to "train".

    Returns:
      dict: slice data class distribution.

    """

    # Calculate class balance
    slice_distribution = {}

    # Add base slice if needed
    if "base" not in slice_func_dict.keys():
        slice_func_dict["base"] = base_slice

    for dataloader in dataloaders:
        labels = dataloader.dataset.Y_dict[  # type: ignore
            dataloader.task_to_label_dict[task.name]
        ]
        for slice_name, slice_func in slice_func_dict.items():
            indicators = slice_func(dataloader.dataset)
            slice_ind_name = f"{task.name}_slice:ind_{slice_name}"
            slice_pred_name = f"{task.name}_slice:pred_{slice_name}"

            pred_labels = indicators * labels
            ind_labels = indicators
            ind_labels[ind_labels == 0] = 2

            if dataloader.split == split and slice_name != "base":
                ind_classes, ind_counts = np.unique(
                    ind_labels.numpy(), return_counts=True
                )
                if ind_classes.shape[0] == 2:
                    slice_distribution[slice_ind_name] = torch.Tensor(  # type: ignore
                        np.sum(ind_counts) / ind_counts / ind_classes.shape[0]
                    )
                pred_classes, pred_counts = np.unique(
                    pred_labels.numpy(), return_counts=True
                )
                if (pred_classes[0] == 0 and pred_classes.shape[0] == 3) or (
                    pred_classes[0] == 1 and pred_classes.shape[0] == 2
                ):
                    if pred_classes[0] == 0:
                        slice_distribution[
                            slice_pred_name
                        ] = torch.Tensor(  # type: ignore
                            1 - pred_counts[1:] / np.sum(pred_counts[1:])
                        )
                    else:
                        slice_distribution[
                            slice_pred_name
                        ] = torch.Tensor(  # type: ignore
                            1 - pred_counts / np.sum(pred_counts)
                        )

            # Update slice indicator and predictor labels
            dataloader.dataset.Y_dict.update(  # type: ignore
                {slice_ind_name: ind_labels, slice_pred_name: pred_labels}
            )
            # Update dataloader task_to_label_dict
            dataloader.task_to_label_dict.update(
                {slice_ind_name: slice_ind_name, slice_pred_name: slice_pred_name}
            )
        msg = (
            f"Loaded slice labels for task {task.name}, slice {slice_name}, "
            f"split {dataloader.split}."
        )
        logger.info(msg)

    return slice_distribution


@slicing_function()
def base_slice(example: Any) -> bool:
    r"""Base slice which always to return True.

    Args:
      example(Any): Sample to check if it's in the slice or not.

    Returns:
      bool: True

    """
    return True
