import random
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def set_random_seed(seed: int = None) -> None:
    r"""Set random seed for random, numpy, and pytorch.

    Args:
      seed(int): The random seed, defaults to none.

    """
    if seed is not None:
        seed = int(seed)

    # Set random seed for random
    random.seed(seed)
    # Set random seed for all numpy operations
    np.random.seed(seed=seed)

    # Set random seed for PyTorch
    if isinstance(seed, int):
        torch.manual_seed(seed)


def list_to_tensor(
    item_list: List[Tensor], min_len: int = 0, max_len: int = 0
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Convert the list of torch.Tensor into a torch.Tensor.

    Args:
      item_list(List[Tensor]): The tensor for converting.
      min_len(int): Min length of sequence of data, defaults to 0.
      max_len(int): Max length of sequence of data, defaults to 0.

    Returns:
      Tuple[Tensor, Optional[Tensor]]: The converted tensor and the corresponding
        mask tensor.

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
    r"""Convert the batch into a padded tensor and mask tensor.

    Args:
      batch(List[Tensor]): The tensor for padding.
      min_len(int): Min length of sequence of padding, defaults to 0.
      max_len(int): Max length of sequence of padding, defaults to 0.
      pad_value(int): The value to use for padding, defaults to 0.
      left_padding(boolean): If True, pad on the left, otherwise on the right,
        defaults to False.

    Returns:
      Tuple[Tensor, Tensor]: The padded tensor and corresponding mask tensor.

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
    r"""Identify the class with the maximum probability.

    Args:
      probs(ndarray): predicted probabilities.

    Returns:
      ndarray: predicted labels.

    """

    return np.argmax(probs, axis=-1)


def pred_to_prob(preds: ndarray, n_classes: int) -> ndarray:
    r"""Converts predicted labels to probabilistic labels.

    Args:
      preds(ndarray): Predicted labels.
      n_classes(int): Total number of classes.

    Returns:
      ndarray: predicted probabilities.

    """

    preds = preds.reshape(-1)
    probs = np.zeros((preds.shape[0], n_classes))

    for idx, class_idx in enumerate(preds):
        probs[idx, class_idx] = 1.0

    return probs


def move_to_device(obj: Any, device: Optional[int] = -1) -> Any:
    r"""Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
      to the specified GPU (or do nothing, if they should beon the CPU).

        device = -1 -> "cpu"
        device =  0 -> "cuda:0"

      Originally from:
        https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py

    Args:
      obj(Any): The object to convert.
      device(int): The device id, defaults to -1.

    Returns:
      Any: The converted object.

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
    array: Union[ndarray, List[Any], Tensor], flatten: bool = False
) -> ndarray:
    r"""Covert an array to a numpy array.

    Args:
      array(ndarray or list or Tensor): An array to convert.
      flatten(bool): Whether to flatten or not.

    Returns:
      ndarray: Converted array.

    """

    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, list):
        array = np.array(array)
    elif isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    else:
        raise ValueError(f"Unrecognized type {type(array)} to convert to ndarray")

    if flatten:
        array = array.reshape(-1)

    return array


def merge(
    x: Dict[str, Any], y: Dict[str, Any], specical_keys: Union[str, List[str]] = None
) -> Dict[str, Any]:
    r"""Merge two nested dictionaries. Overwrite values in x with values in y.

    Args:
      x(dict): The original dict.
      y(dict): The new dict.
      specical_keys(str or list of str): The specical keys to replace
        instead of merging, defaults to None.

    Returns:
      dict: The updated dic.

    """

    if x is None:
        return y
    if y is None:
        return x

    if isinstance(specical_keys, str):
        specical_keys = [specical_keys]

    merged = {**x, **y}

    xkeys = x.keys()

    for key in xkeys:
        if specical_keys is not None and key in specical_keys and key in y:
            merged[key] = y[key]
        elif isinstance(x[key], dict) and key in y:
            merged[key] = merge(x[key], y[key], specical_keys)

    return merged


def str2bool(v: str) -> bool:
    r"""Parse str to bool.

    Args:
      v(str): The string to parse.

    Returns:
      bool: The parsed value.

    """

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2dict(v: str) -> Dict[str, str]:
    r"""Parse str to dict.

    Args:
      v(str): The string to parse.

    Returns:
      dict: The parsed dict.

    """

    dict = {}
    for token in v.split(","):
        key, value = token.split(":")
        dict[key] = value

    return dict


def str2list(v: str, delim: str = ",") -> List[str]:
    r"""Parse str to list.

    Args:
      v(str): The string to parse.
      delim(str): The delimiter used to split string.

    Returns:
      list: The parsed list.

    """

    return [t.strip() for t in v.split(delim)]


def nullable_float(v: str) -> Optional[float]:
    r"""Parse string to nullable float.

    Args:
      v(str): The string to parse.

    Returns:
      float or None: The parsed value.

    """
    if not v or v.lower() in ["none", "null"]:
        return None
    return float(v)


def nullable_int(v: str) -> Optional[int]:
    r"""Parse string to nullable int.

    Args:
      v(str): The string to parse.

    Returns:
      int or None: The parsed value.

    """
    if not v or v.lower() in ["none", "null"]:
        return None
    return int(v)


def nullable_string(v: str) -> Optional[str]:
    r"""Parse string to nullable string.

    Args:
      v(str): The string to parse.

    Returns:
      str or None: The parsed value.

    """
    if not v or v.lower() in ["none", "null"]:
        return None
    return v


def construct_identifier(
    task_name: str, data_name: str, split_name: str, metric_name: Optional[str] = None
) -> str:
    r"""Construct identifier.

    Args:
      task_name(str): Task name.
      data_name(str): Data set name.
      split_name(str): Split name.
      metric_name(str): Metric name, defaults to None.

    Returns:
      str: The identifier.

    """

    if metric_name:
        return f"{task_name}/{data_name}/{split_name}/{metric_name}"
    else:
        return f"{task_name}/{data_name}/{split_name}"


def random_string(length: int = 5) -> str:
    r"""Generate a random string of fixed length.

    Args:
      length(int): The length of random string, defaults to 5.

    Returns:
      str: The random string.

    """

    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))
