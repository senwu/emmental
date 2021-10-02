"""Emmental dataset and dataloader."""
import copy
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from emmental.meta import Meta
from emmental.utils.utils import list_to_tensor, random_string

logger = logging.getLogger(__name__)


class EmmentalDataset(Dataset):
    """Emmental dataset.

    An advanced dataset class to handle that the input data contains multiple fields
    and the output data contains multiple label sets.

    Args:
      name: The name of the dataset.
      X_dict: The feature dict where key is the feature name and value is the
        feature.
      Y_dict: The label dict where key is the label name and value is
        the label, defaults to None.
      uid: The unique id key in the X_dict, defaults to None.
    """

    def __init__(
        self,
        name: str,
        X_dict: Dict[str, Any],
        Y_dict: Optional[Dict[str, Tensor]] = None,
        uid: Optional[str] = None,
    ) -> None:
        """Initialize EmmentalDataset."""
        self.name = name
        self.uid = uid
        self.X_dict = X_dict
        self.Y_dict = Y_dict

        if self.uid and self.uid not in self.X_dict:
            raise ValueError(f"Cannot find {self.uid} in X_dict.")

        if self.uid is None:
            self.uid = "_uids_"
            while self.uid in X_dict:
                self.uid = f"_uids_{random_string(3)}_"

            uids = [f"{self.name}_{idx}" for idx in range(self.__len__())]
            self.add_features({f"{self.uid}": uids})

            if Meta.config["meta_config"]["verbose"]:
                logger.info(
                    f"Auto generate uids for dataset {self.name} under {self.uid}."
                )

        # if self.Y_dict is not None:
        #     for name, label in self.Y_dict.items():
        #         if not isinstance(label, Tensor):
        #             raise ValueError(
        #                 f"Label {name} should be torch.Tensor, not {type(label)}."
        #             )

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Tensor]], Dict[str, Any]]:
        """Get item by index.

        Args:
          index: The index of the item.

        Returns:
          Tuple of x_dict and y_dict
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}

        if self.Y_dict is None:
            return x_dict
        else:
            y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        return x_dict, y_dict

    def __len__(self) -> int:
        """Total number of items in the dataset."""
        try:
            return len(next(iter(self.X_dict.values())))
        except StopIteration:
            return 0

    def _update_dict(self, ori_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> None:
        """Update original dict with new dict.

        Args:
          ori_dict: The original dict.
          new_dict: The new dict.
        """
        for key, value in new_dict.items():
            ori_dict[key] = value

    def _remove_key(self, ori_dict: Dict[str, Any], key: str) -> None:
        """Remove key from dataset dict.

        Args:
          ori_dict: The original dict.
          key: The key to remove from the original dict.
        """
        if key in ori_dict:
            del ori_dict[key]

    def add_features(self, X_dict: Dict[str, Any]) -> None:
        """Add new features into X_dict.

        Args:
          X_dict: The new feature dict to add into the existing feature dict.
        """
        self._update_dict(self.X_dict, X_dict)

    def add_labels(self, Y_dict: Dict[str, Tensor]) -> None:
        """Add new labels into Y_dict.

        Args:
          Y_dict: the new label dict to add into the existing label dict
        """
        for name, label in Y_dict.items():
            if not isinstance(label, Tensor):
                raise ValueError(f"Label {name} should be torch.Tensor.")

        if self.Y_dict is None:
            self.Y_dict = {}
        self._update_dict(self.Y_dict, Y_dict)

    def remove_feature(self, feature_name: str) -> None:
        """Remove one feature from feature dict.

        Args:
          feature_name: The feature that removes from feature dict.
        """
        self._remove_key(self.X_dict, feature_name)

    def remove_label(self, label_name: str) -> None:
        """Remove one label from label dict.

        Args:
          label_name: The label that removes from label dict.
        """
        self._remove_key(self.Y_dict, label_name)


def emmental_collate_fn(
    batch: Union[List[Tuple[Dict[str, Any], Dict[str, Tensor]]], List[Dict[str, Any]]]
) -> Union[Tuple[Dict[str, Any], Dict[str, Tensor]], Dict[str, Any]]:
    """Collate function.

    Args:
      batch: The batch to collate.

    Returns:
      The collated batch.
    """
    X_batch: defaultdict = defaultdict(list)
    Y_batch: defaultdict = defaultdict(list)

    for item in batch:
        # Check if batch is (x_dict, y_dict) pair
        if isinstance(item, dict):
            x_dict = item
            y_dict: Dict[str, Any] = {}
        else:
            x_dict, y_dict = item
        for field_name, value in x_dict.items():
            if isinstance(value, list):
                X_batch[field_name] += value
            else:
                X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            if isinstance(value, list):
                Y_batch[label_name] += value
            else:
                Y_batch[label_name].append(value)

    field_names = copy.deepcopy(list(X_batch.keys()))

    for field_name in field_names:
        values = X_batch[field_name]
        # Only merge list of tensors
        if isinstance(values[0], Tensor):
            item_tensor, item_mask_tensor = list_to_tensor(
                values,
                min_len=Meta.config["data_config"]["min_data_len"],
                max_len=Meta.config["data_config"]["max_data_len"],
            )
            X_batch[field_name] = item_tensor
            if item_mask_tensor is not None:
                X_batch[f"{field_name}_mask"] = item_mask_tensor

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(
            values,
            min_len=Meta.config["data_config"]["min_data_len"],
            max_len=Meta.config["data_config"]["max_data_len"],
        )[0]

    if len(Y_batch) != 0:
        return dict(X_batch), dict(Y_batch)
    else:
        return dict(X_batch)


class EmmentalDataLoader(DataLoader):
    """Emmental dataLoader.

    An advanced dataloader class which contains mapping from task to label (which
    label(s) to use in dataset's Y_dict for this task), and split (which part this
    dataset belongs to) information.

    Args:
      task_to_label_dict: The task to label mapping where key is the task name
        and value is the label(s) for that task and should be the key in Y_dict.
      dataset: The dataset to construct the dataloader
      split: The split information, defaults to "train".
      collate_fn: The function that merges a list of samples to
        form a mini-batch, defaults to emmental_collate_fn.
      n_batches: Total number of batches.
      **Kwargs: Other arguments of dataloader.
    """

    def __init__(
        self,
        task_to_label_dict: Dict[str, str],
        dataset: EmmentalDataset,
        split: str = "train",
        collate_fn: Callable = emmental_collate_fn,
        n_batches: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EmmentalDataLoader."""
        assert isinstance(
            dataset, EmmentalDataset
        ), "dataset should inherent from EmmentalDataset."
        assert isinstance(
            task_to_label_dict, dict
        ), "task_to_label_dict should be a dict."

        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

        self.task_to_label_dict = task_to_label_dict
        self.data_name = dataset.name
        self.uid = dataset.uid
        self.split = split
        self.n_batches = n_batches

        for task_name, label_names in task_to_label_dict.items():
            if label_names is None:
                continue
            if not isinstance(label_names, list):
                label_names = [label_names]  # type: ignore
            if not isinstance(dataset[0], dict):
                unrecognized_labels = set(label_names) - set(list(dataset[0][1].keys()))
            else:
                unrecognized_labels = set(label_names)
            if len(unrecognized_labels) > 0:
                msg = (
                    f"Unrecognized Label {unrecognized_labels} of Task "
                    f"{task_name} in dataset {dataset.name}."
                )
                logger.error(msg)
                raise ValueError(msg)
