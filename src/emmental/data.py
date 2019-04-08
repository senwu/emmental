from collections import defaultdict

from emmental.utils.utils import list_to_tensor
from torch.utils.data import DataLoader, Dataset


class EmmentalDataset(Dataset):
    """A advanced dataset class to handle that the input data contains mulitple
    fields and the output data contains multiple label sets

    :param X_dict: the input dict where key is the field name and value is the
    data
    :type X_dict: dict
    :param Y_dict: the output dict where key is the label set name and value is
    the label
    :type Y_dict: dict
    """

    def __init__(self, X_dict, Y_dict):
        self.X_dict = X_dict
        self.Y_dict = Y_dict

    def __getitem__(self, index):
        x_dict = {key: field[index] for key, field in self.X_dict.items()}
        y_dict = {key: label[index] for key, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        return len(next(iter(self.X_dict.values())))


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

    return X_batch, Y_batch


class EmmentalDataLoader(DataLoader):
    def __init__(self, dataset, collate_fn=emmental_collate_fn, **kwargs):
        assert isinstance(dataset, EmmentalDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
