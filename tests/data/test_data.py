#! /usr/bin/env python
import logging

import torch

from emmental.data import EmmentalDataLoader, EmmentalDataset


def test_emmental_dataset(caplog):
    """Unit test of emmental dataset"""

    caplog.set_level(logging.INFO)

    x1 = [
        torch.Tensor([1]),
        torch.Tensor([1, 2]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3, 4, 5]),
    ]

    y1 = [
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
    ]

    dataset = EmmentalDataset({"data1": x1}, {"label1": y1})

    # Check if the dataset is correctly constructed
    assert torch.equal(dataset[0][0]["data1"], torch.Tensor([1]))
    assert torch.equal(dataset[0][1]["label1"], torch.Tensor([0]))

    x2 = [
        torch.Tensor([1, 2, 3, 4, 5]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2]),
        torch.Tensor([1]),
    ]

    dataset.add_features({"data2": x2})

    # Check add one more feature to dataset
    assert torch.equal(dataset[0][0]["data2"], torch.Tensor([1, 2, 3, 4, 5]))

    y2 = [
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
    ]

    dataset.add_labels({"label2": y2})

    # Check add one more label to dataset
    assert torch.equal(dataset[0][1]["label2"], torch.Tensor([1]))

    dataset.remove_label("label1")

    # Check remove one more label to dataset
    assert "label1" not in dataset.Y_dict


def test_emmental_dataloader(caplog):
    """Unit test of emmental dataloader"""

    caplog.set_level(logging.INFO)

    x1 = [
        torch.Tensor([1]),
        torch.Tensor([1, 2]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3, 4, 5]),
    ]

    y1 = [
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
        torch.Tensor([0]),
    ]

    x2 = [
        torch.Tensor([1, 2, 3, 4, 5]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2]),
        torch.Tensor([1]),
    ]

    y2 = [
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
        torch.Tensor([1]),
    ]

    dataset = EmmentalDataset({"data1": x1, "data2": x2}, {"label1": y1, "label2": y2})

    dataloader1 = EmmentalDataLoader(dataset, batch_size=2)

    x_batch, y_batch = next(iter(dataloader1))

    # Check if the dataloader is correctly constructed
    assert torch.equal(x_batch["data1"], torch.Tensor([[1, 0], [1, 2]]))
    assert torch.equal(
        x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
    )
    assert torch.equal(y_batch["label1"], torch.Tensor([[0], [0]]))
    assert torch.equal(y_batch["label2"], torch.Tensor([[1], [1]]))

    dataloader2 = EmmentalDataLoader(dataset, batch_size=3)

    x_batch, y_batch = next(iter(dataloader2))

    # Check if the dataloader with differet batch size is correctly constructed
    assert torch.equal(
        x_batch["data1"], torch.Tensor([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
    )
    assert torch.equal(
        x_batch["data2"],
        torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
    )
    assert torch.equal(y_batch["label1"], torch.Tensor([[0], [0], [0]]))
    assert torch.equal(y_batch["label2"], torch.Tensor([[1], [1], [1]]))

    y3 = [
        torch.Tensor([2]),
        torch.Tensor([2]),
        torch.Tensor([2]),
        torch.Tensor([2]),
        torch.Tensor([2]),
    ]

    dataset.Y_dict["label2"] = y3

    x_batch, y_batch = next(iter(dataloader1))
    # Check dataloader is correctly updated with update dataset
    assert torch.equal(
        x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
    )
    assert torch.equal(y_batch["label2"], torch.Tensor([[2], [2]]))

    x_batch, y_batch = next(iter(dataloader2))
    assert torch.equal(
        x_batch["data2"],
        torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
    )
    assert torch.equal(y_batch["label2"], torch.Tensor([[2], [2], [2]]))
