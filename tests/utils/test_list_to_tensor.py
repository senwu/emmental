import logging

import torch

from emmental.utils.utils import list_to_tensor


def test_list_to_tensor(caplog):
    """Unit test of list to tensor"""

    caplog.set_level(logging.INFO)

    # list of 1-D tensor with the different length
    batch = [torch.Tensor([1, 2]), torch.Tensor([3]), torch.Tensor([4, 5, 6])]

    padded_batch = list_to_tensor(batch)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]]))

    # list of 1-D tensor with the same length
    batch = [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])]

    padded_batch = list_to_tensor(batch)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # list of 2-D tensor with the same size
    batch = [
        torch.Tensor([[1, 2, 3], [1, 2, 3]]),
        torch.Tensor([[4, 5, 6], [4, 5, 6]]),
        torch.Tensor([[7, 8, 9], [7, 8, 9]]),
    ]

    padded_batch = list_to_tensor(batch)

    assert torch.equal(
        padded_batch,
        torch.Tensor(
            [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]], [[7, 8, 9], [7, 8, 9]]]
        ),
    )

    # list of tensor with the different size
    batch = [
        torch.Tensor([[1, 2], [2, 3]]),
        torch.Tensor([4, 5, 6]),
        torch.Tensor([7, 8, 9, 0]),
    ]

    padded_batch = list_to_tensor(batch)

    assert torch.equal(
        padded_batch, torch.Tensor([[1, 2, 2, 3], [4, 5, 6, 0], [7, 8, 9, 0]])
    )
