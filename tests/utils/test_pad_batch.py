import logging

import torch

from emmental.utils.utils import pad_batch


def test_pad_batch(caplog):
    """Unit test of pad batch"""

    caplog.set_level(logging.INFO)

    batch = [torch.Tensor([1, 2]), torch.Tensor([3]), torch.Tensor([4, 5, 6])]
    padded_batch, mask_batch = pad_batch(batch)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]]))
    assert torch.equal(
        mask_batch, mask_batch.new_tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    )

    padded_batch, mask_batch = pad_batch(batch, max_len=2)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2], [3, 0], [4, 5]]))
    assert torch.equal(mask_batch, mask_batch.new_tensor([[0, 0], [0, 1], [0, 0]]))

    padded_batch, mask_batch = pad_batch(batch, pad_value=-1)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2, -1], [3, -1, -1], [4, 5, 6]]))
    assert torch.equal(
        mask_batch, mask_batch.new_tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    )

    padded_batch, mask_batch = pad_batch(batch, left_padded=True)

    assert torch.equal(padded_batch, torch.Tensor([[0, 1, 2], [0, 0, 3], [4, 5, 6]]))
    assert torch.equal(
        mask_batch, mask_batch.new_tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    )

    padded_batch, mask_batch = pad_batch(batch, max_len=2, left_padded=True)

    assert torch.equal(padded_batch, torch.Tensor([[1, 2], [0, 3], [5, 6]]))
    assert torch.equal(mask_batch, mask_batch.new_tensor([[0, 0], [1, 0], [0, 0]]))
