import random

import numpy as np
import torch


def set_random_seed(seed):
    """Set random seed."""

    seed = int(seed)

    # Set random seed for random
    random.seed(seed)
    # Set random seed for all numpy operations
    np.random.seed(seed=seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def list_to_tensor(item_list):
    """Convert the list of items into a tensor."""

    # Convert 2 or more-D tensor with the same shape
    if all(
        (item_list[i].size() == item_list[0].size()) and (len(item_list[i].size()) != 1)
        for i in range(len(item_list))
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch([item.view(-1) for item in item_list])

    return item_tensor


def pad_batch(batch, max_len=0, pad_value=0, left_padded=False):
    """Convert the batch into a padded tensor and mask tensor.

    :param batch: The data for padding.
    :type batch: list of torch.Tensor
    :param max_len: Max length of sequence of padding.
    :type max_len: int
    :param pad_value: The value to use for padding
    :type pad_value: int
    :param left_padding: if True, pad on the left, otherwise on the right.
    :type left_padding: boolean
    :return: The padded matrix and correspoing mask matrix.
    :rtype: pair of torch.Tensors with shape (batch_size, max_seq_len)
    """

    batch_size = len(batch)
    max_seq_len = int(np.max([len(item) for item in batch]))

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(len(item), max_seq_len)
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value).type_as(
        padded_batch
    )

    return padded_batch, mask_batch


def prob_to_pred(probs):
    """Identify the class with the maximum probability (add 1 since we assume label
    class starts from 1)

    :param probs: probabilities
    :type probs: np.array
    """

    return np.argmax(probs, axis=-1) + 1
