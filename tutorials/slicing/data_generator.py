import random
from collections import defaultdict

import numpy as np
import torch


def get_circle_mask(X, center, radius):
    h, k = center
    mask = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < radius
    return mask.astype(np.bool)


def generate_data(N, label_flips):
    """ Generates data in numpy form.

    Returns: (
        [uids_train, uids_val, uids_test],
        [X_train, X_val, X_test],
        [Y_train, Y_val, Y_test]
    )
    """

    uids = list(range(N))
    X = np.random.random((N, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    # abberation in decision boundary
    for mask_fn, label in label_flips.items():
        Y[mask_fn(X)] = label

    uid_lists, Xs, Ys = split_data(uids, X, Y, splits=[0.5, 0.25, 0.25], shuffle=True)
    return uid_lists, Xs, Ys


def generate_slice_labels(X, Y, slice_funcs, create_ind, create_preds):
    """
    Args:
        X: [N x D] data
        Y: [N x 1] labels in {0, 1}
        slice_funcs [dict]: mapping slice_names to slice_fn(X),
            which returns [N x 1] boolean mask indic. whether examples are in slice
        create_ind [bool]: indicating whether we should create indicator labels

    Returns:
        slice_labels [dict]: mapping slice_names to dict of {
            pred: [N x 1] in {0, 1, 2} original Y abstaining (with 0)
                on examples not in slice
            ind: [N x 1] in {1, 2} mask labels in categorical format
        }
    """
    slice_labels = {}
    for slice_name, slice_fn in slice_funcs.items():
        slice_mask = slice_fn(X)
        Y_gt = Y.copy()
        # if not in slice, abstain with label = 0
        Y_gt[np.logical_not(slice_mask)] = 0

        #         # convert from True/False mask -> 1,2 categorical labels
        #         categorical_indicator = convert_labels(
        #             slice_mask.astype(np.int), "onezero", "categorical"
        #         )
        slice_labels[slice_name] = {}
        if create_preds:
            slice_labels[slice_name] = Y_gt
    #         if create_ind:
    #             slice_labels[slice_name].update({"ind": categorical_indicator})

    return slice_labels


def convert_labels(Y, source, target):
    """Convert a matrix from one label type to another
    Args:
        Y: A np.ndarray or torch.Tensor of labels (ints) using source convention
        source: The convention the labels are currently expressed in
        target: The convention to convert the labels to
    Returns:
        Y: an np.ndarray or torch.Tensor of labels (ints) using the target convention
    Conventions:
        'categorical': [0: abstain, 1: positive, 2: negative]
        'plusminus': [0: abstain, 1: positive, -1: negative]
        'onezero': [0: negative, 1: positive]
    Note that converting to 'onezero' will combine abstain and negative labels.
    """
    if Y is None:
        return Y
    if isinstance(Y, np.ndarray):
        Y = Y.copy()
        assert Y.dtype == np.int64
    elif isinstance(Y, torch.Tensor):
        Y = Y.clone()
        assert np.sum(Y.cpu().numpy() - Y.cpu().numpy().astype(int)) == 0.0
    else:
        raise ValueError("Unrecognized label data type.")
    negative_map = {"categorical": 2, "plusminus": -1, "onezero": 0}
    Y[Y == negative_map[source]] = negative_map[target]
    return Y


def split_data(
    *inputs,
    splits=[0.5, 0.5],
    shuffle=True,
    stratify_by=None,
    index_only=False,
    seed=None,
):
    """Splits inputs into multiple splits of defined sizes
    Args:
        inputs: correlated tuples/lists/arrays/matrices/tensors to split
        splits: list containing split sizes (fractions or counts);
        shuffle: if True, shuffle the data before splitting
        stratify_by: (None or an input) if not None, use these labels to
            stratify the splits (separating the data into groups by these
            labels and sampling from those, rather than from the population at
            large); overrides shuffle
        index_only: if True, return only the indices of the new splits, not the
            split data itself
        seed: (int) random seed
    Example usage:
        Ls, Xs, Ys = split_data(L, X, Y, splits=[0.8, 0.1, 0.1])
        OR
        assignments = split_data(Y, splits=[0.8, 0.1, 0.1], index_only=True)
    Note: This is very similar to scikit-learn's train_test_split() method,
        but with support for more than two splits.
    """

    def fractions_to_counts(fracs, n):
        """Converts a list of fractions to a list of counts that sum to n"""
        counts = [int(np.round(n * frac)) for frac in fracs]
        # Ensure sum of split counts sums to n
        counts[-1] = n - sum(counts[:-1])
        return counts

    def slice_data(data, indices):
        if isinstance(data, list) or isinstance(data, tuple):
            return [d for i, d in enumerate(data) if i in set(indices)]
        else:
            try:
                # Works for np.ndarray, scipy.sparse, torch.Tensor
                return data[indices]
            except TypeError:
                raise Exception(
                    f"split_data() currently only accepts inputs "
                    f"of type tuple, list, np.ndarray, scipy.sparse, or "
                    f"torch.Tensor; not {type(data)}"
                )

    # Setting random seed
    if seed is not None:
        random.seed(seed)

    try:
        n = len(inputs[0])
    except TypeError:
        n = inputs[0].shape[0]
    num_splits = len(splits)

    # Check splits for validity and convert to fractions
    if all(isinstance(x, int) for x in splits):
        if not sum(splits) == n:
            raise ValueError(
                f"Provided split counts must sum to n ({n}), not {sum(splits)}."
            )
        fracs = [count / n for count in splits]

    elif all(isinstance(x, float) for x in splits):
        if not sum(splits) == 1.0:
            raise ValueError(f"Split fractions must sum to 1.0, not {sum(splits)}.")
        fracs = splits

    else:
        raise ValueError("Splits must contain all ints or all floats.")

    # Make sampling pools
    if stratify_by is None:
        pools = [np.arange(n)]
    else:
        pools = defaultdict(list)
        for i, val in enumerate(stratify_by):
            pools[val].append(i)
        pools = list(pools.values())

    # Make index assignments
    assignments = [[] for _ in range(num_splits)]
    for pool in pools:
        if shuffle or stratify_by is not None:
            random.shuffle(pool)

        counts = fractions_to_counts(fracs, len(pool))
        counts.insert(0, 0)
        cum_counts = np.cumsum(counts)
        for i in range(num_splits):
            assignments[i].extend(pool[cum_counts[i] : cum_counts[i + 1]])

    if index_only:
        return assignments
    else:
        outputs = []
        for data in inputs:
            data_splits = []
            for split in range(num_splits):
                data_splits.append(slice_data(data, assignments[split]))
            outputs.append(data_splits)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
