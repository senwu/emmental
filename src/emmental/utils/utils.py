import random
import string

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


def list_to_tensor(item_list, min_len=0, max_len=0):
    """Convert the list of torch.Tensor into a torch.Tensor."""

    # Convert single value tensor
    if all(item_list[i].dim() == 0 for i in range(len(item_list))):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert 2 or more-D tensor with the same shape
    elif all(
        (item_list[i].size() == item_list[0].size()) and (len(item_list[i].size()) != 1)
        for i in range(len(item_list))
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch(
            [item.view(-1) for item in item_list], min_len, max_len
        )

    return item_tensor


def pad_batch(batch, min_len=0, max_len=0, pad_value=0, left_padded=False):
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

    max_seq_len = max(max_seq_len, min_len)

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


def pred_to_prob(preds, n_classes):
    """Converts predicted labels to probabilistic labels

    :param preds: predicted labels
    :type probs: np.array
    """

    preds = preds.reshape(-1)
    probs = np.zeros((preds.shape[0], n_classes))

    for idx, class_idx in enumerate(preds):
        probs[idx, class_idx - 1] = 1.0

    return probs


def move_to_device(obj, device=-1):
    """
    Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
    to the specified GPU (or do nothing, if they should beon the CPU).
    device = -1 -> "cpu"
    device =  0 -> "cuda:0"
    Originally from:
    https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py
    """

    if device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def array_to_numpy(array, flatten=False):
    """
    Covert an array to a numpy array.

    :param array: An array to convert
    :type array: list or np.ndarray
    :param flatten: Whether to flatten or not
    :type flatten: bool
    :return: Converted np.ndarray
    :rtype: np.ndarray
    """

    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, list):
        array = np.array(array)
    elif isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    else:
        raise ValueError(f"Unrecognized type {type(array)} to convert to np.ndarray")

    if flatten:
        array = array.reshpae(-1)

    return array


def merge(x, y):
    """Merge two nested dictionaries. Overwrite values in x with values in y."""

    merged = {**x, **y}

    xkeys = x.keys()

    for key in xkeys:
        if isinstance(x[key], dict) and key in y:
            merged[key] = merge(x[key], y[key])

    return merged


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2dict(v):
    dict = {}
    for token in v.split(","):
        key, value = token.split(":")
        dict[key] = value

    return dict


def str2list(v, delim=","):
    return [t.strip() for t in v.split(delim)]


def nullable_string(v):
    if not v or v.lower() in ["none", "null"]:
        return None
    return v


def construct_identifier(task_name, data_name, split_name, metric_name=None):
    if metric_name:
        return f"{task_name}/{data_name}/{split_name}/{metric_name}"
    else:
        return f"{task_name}/{data_name}/{split_name}"


def random_string(length=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))
