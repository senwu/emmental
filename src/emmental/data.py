from torch.utils.data import DataLoader, Dataset


class EmmentalDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def emmental_collate_fn(batch):
    pass


class EmmentalDataLoader(DataLoader):
    def __init__(self, dataset, collate_fn=emmental_collate_fn):
        assert isinstance(dataset, EmmentalDataset)
        super().__init__(dataset, collate_fn=collate_fn)
