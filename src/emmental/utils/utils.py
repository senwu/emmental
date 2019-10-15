import random
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def set_random_seed(seed: int) -> None:
    """Set random seed."""

    seed = int(seed)

    # Set random seed for random
    random.seed(seed)
    # Set random seed for all numpy operations
    np.random.seed(seed=seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)


def list_to_tensor(
    item_list: List[Tensor], min_len: int = 0, max_len: int = 0
) -> Union[Tuple[Tensor, None], Tuple[Tensor, Tensor]]:
    """Convert the list of torch.Tensor into a torch.Tensor.

    :param item_list: The data for converting.
    :type batch: list of torch.Tensor
    :param min_len: Min length of sequence of data.
    :type min_len: int
    :param max_len: Max length of sequence of data.
    :type max_len: int
    :return: The converted tensor and the correspoing mask tensor.
    :rtype: tuple of torch.Tensors with shape (batch_size, max_seq_len) or None
    """

    item_mask_tensor = None

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
        item_tensor, item_mask_tensor = pad_batch(
            [item.view(-1) for item in item_list], min_len, max_len
        )

    return item_tensor, item_mask_tensor


def pad_batch(
    batch: List[Tensor],
    min_len: int = 0,
    max_len: int = 0,
    pad_value: int = 0,
    left_padded: bool = False,
) -> Tuple[Tensor, Tensor]:
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
    :rtype: tuple of torch.Tensors with shape (batch_size, max_seq_len)
    """
    batch_size = len(batch)
    max_seq_len = int(np.max([item.size()[0] for item in batch]))

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    max_seq_len = max(max_seq_len, min_len)

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(item.size()[0], max_seq_len)
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value)

    return padded_batch, mask_batch


def prob_to_pred(probs: ndarray) -> ndarray:
    """Identify the class with the maximum probability (add 1 since we assume label
    class starts from 1)

    :param probs: probabilities
    :type probs: np.array
    """

    return np.argmax(probs, axis=-1) + 1


def pred_to_prob(preds: ndarray, n_classes: int) -> ndarray:
    """Converts predicted labels to probabilistic labels

    :param preds: predicted labels
    :type probs: np.array
    """

    preds = preds.reshape(-1)
    probs = np.zeros((preds.shape[0], n_classes))

    for idx, class_idx in enumerate(preds):
        probs[idx, class_idx - 1] = 1.0

    return probs


def move_to_device(obj, device: Optional[int] = -1):  # type: ignore
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
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def array_to_numpy(
    array: Union[ndarray, List[Union[ndarray, int, float]]], flatten: bool = False
) -> ndarray:
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


def merge(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two nested dictionaries. Overwrite values in x with values in y."""

    merged = {**x, **y}

    xkeys = x.keys()

    for key in xkeys:
        if isinstance(x[key], dict) and key in y:
            merged[key] = merge(x[key], y[key])

    return merged


def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2dict(v: str) -> Dict[str, str]:
    dict = {}
    for token in v.split(","):
        key, value = token.split(":")
        dict[key] = value

    return dict


def str2list(v: str, delim: str = ",") -> List[str]:
    return [t.strip() for t in v.split(delim)]


def nullable_string(v: str) -> Optional[str]:
    if not v or v.lower() in ["none", "null"]:
        return None
    return v


def construct_identifier(
    task_name: str, data_name: str, split_name: str, metric_name: Optional[str] = None
) -> str:
    if metric_name:
        return f"{task_name}/{data_name}/{split_name}/{metric_name}"
    else:
        return f"{task_name}/{data_name}/{split_name}"


def random_string(length: int = 5) -> str:
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))
