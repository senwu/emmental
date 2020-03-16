import logging

import pytest
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

    y1 = torch.Tensor([0, 0, 0, 0, 0])

    dataset = EmmentalDataset(
        X_dict={"data1": x1}, Y_dict={"label1": y1}, name="new_data"
    )

    # Check if the dataset is correctly constructed
    assert torch.equal(dataset[0][0]["data1"], x1[0])
    assert torch.equal(dataset[0][1]["label1"], y1[0])

    x2 = [
        torch.Tensor([1, 2, 3, 4, 5]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2]),
        torch.Tensor([1]),
    ]

    dataset.add_features(X_dict={"data2": x2})

    dataset.remove_feature("data2")
    assert "data2" not in dataset.X_dict

    dataset.add_features(X_dict={"data2": x2})

    # Check add one more feature to dataset
    assert torch.equal(dataset[0][0]["data2"], x2[0])

    y2 = torch.Tensor([1, 1, 1, 1, 1])

    dataset.add_labels(Y_dict={"label2": y2})

    with pytest.raises(ValueError):
        dataset.add_labels(Y_dict={"label2": x2})

    # Check add one more label to dataset
    assert torch.equal(dataset[0][1]["label2"], y2[0])

    dataset.remove_label(label_name="label1")

    # Check remove one more label to dataset
    assert "label1" not in dataset.Y_dict

    with pytest.raises(ValueError):
        dataset = EmmentalDataset(
            X_dict={"data1": x1}, Y_dict={"label1": y1}, name="new_data", uid="ids"
        )

    dataset = EmmentalDataset(
        X_dict={"_uids_": x1}, Y_dict={"label1": y1}, name="new_data"
    )

    with pytest.raises(ValueError):
        dataset = EmmentalDataset(
            X_dict={"data1": x1}, Y_dict={"label1": x1}, name="new_data"
        )

    with pytest.raises(ValueError):
        dataset = EmmentalDataset(
            X_dict={"data1": x1}, Y_dict={"label1": x1}, name="new_data"
        )


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

    y1 = torch.Tensor([0, 0, 0, 0, 0])

    x2 = [
        torch.Tensor([1, 2, 3, 4, 5]),
        torch.Tensor([1, 2, 3, 4]),
        torch.Tensor([1, 2, 3]),
        torch.Tensor([1, 2]),
        torch.Tensor([1]),
    ]

    y2 = torch.Tensor([1, 1, 1, 1, 1])

    dataset = EmmentalDataset(
        X_dict={"data1": x1, "data2": x2},
        Y_dict={"label1": y1, "label2": y2},
        name="new_data",
    )

    dataloader1 = EmmentalDataLoader(
        task_to_label_dict={"task1": "label1"},
        dataset=dataset,
        split="train",
        batch_size=2,
    )

    x_batch, y_batch = next(iter(dataloader1))

    # Check if the dataloader is correctly constructed
    assert dataloader1.task_to_label_dict == {"task1": "label1"}
    assert dataloader1.split == "train"
    assert torch.equal(x_batch["data1"], torch.Tensor([[1, 0], [1, 2]]))
    assert torch.equal(
        x_batch["data2"], torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]])
    )
    assert torch.equal(y_batch["label1"], torch.Tensor([0, 0]))
    assert torch.equal(y_batch["label2"], torch.Tensor([1, 1]))

    dataloader2 = EmmentalDataLoader(
        task_to_label_dict={"task2": "label2"},
        dataset=dataset,
        split="test",
        batch_size=3,
    )

    x_batch, y_batch = next(iter(dataloader2))

    # Check if the dataloader with differet batch size is correctly constructed
    assert dataloader2.task_to_label_dict == {"task2": "label2"}
    assert dataloader2.split == "test"
    assert torch.equal(
        x_batch["data1"], torch.Tensor([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
    )
    assert torch.equal(
        x_batch["data2"],
        torch.Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
    )
    assert torch.equal(y_batch["label1"], torch.Tensor([0, 0, 0]))
    assert torch.equal(y_batch["label2"], torch.Tensor([1, 1, 1]))

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
