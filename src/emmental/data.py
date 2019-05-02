import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from emmental.utils.utils import list_to_tensor

logger = logging.getLogger(__name__)


class EmmentalDataset(Dataset):
    """An advanced dataset class to handle that the input data contains mulitple
    fields and the output data contains multiple label sets

    :param name: the name of the dataset
    :type name: str
    :param X_dict: the feature dict where key is the feature name and value is the
    feature
    :type X_dict: dict
    :param Y_dict: the label dict where key is the label name and value is
    the label
    :type Y_dict: dict
    """

    def __init__(self, name, X_dict, Y_dict):
        self.name = name
        self.X_dict = X_dict
        self.Y_dict = Y_dict

        for name, label in self.Y_dict.items():
            if not isinstance(label, torch.Tensor):
                raise ValueError(f"Label {name} should be torch.Tensor.")

    def __getitem__(self, index):
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        return len(next(iter(self.X_dict.values())))

    def _update_dict(self, ori_dict, new_dict):
        for key, value in new_dict.items():
            ori_dict[key] = value

    def _remove_key(self, ori_dict, key):
        if key in ori_dict:
            del ori_dict[key]

    def add_features(self, X_dict):
        """Add new features into X_dict

        :param X_dict: the new feature dict to add into the existing feature dict
        :type X_dict: dict
        """

        self._update_dict(self.X_dict, X_dict)

    def add_labels(self, Y_dict):
        """Add new labels into Y_dict

        :param Y_dict: the new label dict to add into the existing label dict
        :type Y_dict: dict
        """

        for name, label in Y_dict.items():
            if not isinstance(label, torch.Tensor):
                raise ValueError(f"Label {name} should be torch.Tensor.")

        self._update_dict(self.Y_dict, Y_dict)

    def remove_feature(self, feature_name):
        """Remove one feature from feature dict

        :param feature_name: the feature that removes from feature dict
        :type feature_name: str
        """

        self._remove_key(self.X_dict, feature_name)

    def remove_label(self, label_name):
        """Remove one label from label dict

        :param label_name: the label that removes from label dict
        :type label_name: str
        """

        self._remove_key(self.Y_dict, label_name)


def emmental_collate_fn(batch):

    X_batch = defaultdict(list)
    Y_batch = defaultdict(list)

    for x_dict, y_dict in batch:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)

    for field_name, values in X_batch.items():
        # Only merge list of tensors
        if isinstance(values[0], torch.Tensor):
            X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    return dict(X_batch), dict(Y_batch)


class EmmentalDataLoader(DataLoader):
    """An advanced dataloader class which contains mapping from task to label (which
    label(s) to use in dataset's Y_dict for this task), and split (which part this
    dataset belongs to) information.

    :param task_to_label_dict: the task to label mapping where key is the task name and
    value is the label(s) for that task and should be the key in Y_dict
    :type task_to_label_dict: dict
    :param dataset: the dataset to construct the dataloader
    :type dataset: torch.utils.data.Datasetwe
    :param split: the split information, defaults to "train"
    :param split: str, optional
    :param collate_fn: the function that merges a list of samples to form a
    mini-batch, defaults to emmental_collate_fn
    :param collate_fn: function, optional
    """

    def __init__(
        self,
        task_to_label_dict,
        dataset,
        split="train",
        collate_fn=emmental_collate_fn,
        **kwargs,
    ):

        assert isinstance(dataset, EmmentalDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

        self.task_to_label_dict = task_to_label_dict
        self.data_name = dataset.name
        self.split = split

        for task_name, label_names in task_to_label_dict.items():
            if not isinstance(label_names, list):
                label_names = [label_names]
            unrecognized_labels = set(label_names) - set(dataset.Y_dict.keys())
            if len(unrecognized_labels) > 0:
                msg = (
                    f"Unrecognized Label {unrecognized_labels} of Task {task_name} in "
                    f"dataset {dataset.name}"
                )
                logger.warn(msg)
                raise ValueError(msg)
