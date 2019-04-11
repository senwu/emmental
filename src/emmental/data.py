from collections import defaultdict

from torch.utils.data import DataLoader, Dataset

from emmental.utils.utils import list_to_tensor


class EmmentalDataset(Dataset):
    """An advanced dataset class to handle that the input data contains mulitple
    fields and the output data contains multiple label sets

    :param X_dict: the feature dict where key is the feature name and value is the
    feature
    :type X_dict: dict
    :param Y_dict: the label dict where key is the label name and value is
    the label
    :type Y_dict: dict
    :param name: the name of the dataset
    :type name: dict
    """

    def __init__(self, X_dict, Y_dict, name=None):
        self.name = name
        self.X_dict = X_dict
        self.Y_dict = Y_dict

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
        X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    return dict(X_batch), dict(Y_batch)


class EmmentalDataLoader(DataLoader):
    """An advanced dataloader class which captures task name, label name (which
    label to use in dataset's Y_dict for this task), and split (which part this
    dataset belongs to) information

    :param task_name: the name of task which uses this dataset
    :type task_name: str
    :param dataset: the dataset to construct the dataloader
    :type dataset: torch.utils.data.Datasetwe
    :param label_name: label name for the task
    :param label_name: str
    :param split: the split information, defaults to "train"
    :param split: str, optional
    :param collate_fn: the function that merges a list of samples to form a
    mini-batch, defaults to emmental_collate_fn
    :param collate_fn: function, optional
    """

    def __init__(
        self,
        task_name,
        dataset,
        label_name,
        split="train",
        collate_fn=emmental_collate_fn,
        **kwargs
    ):
        assert isinstance(dataset, EmmentalDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

        self.task_name = task_name
        self.label_name = label_name
        self.split = split
