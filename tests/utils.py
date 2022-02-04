"""Emmental unit tests' utils."""
import math

from numpy import ndarray


def isequal(dict_a, dict_b, precision=1e-10):
    """Check if two dicts are the same."""
    for key in dict_a:
        if key not in dict_b:
            return False
        if isinstance(dict_a[key], ndarray) and isinstance(dict_b[key], ndarray):
            if not (dict_a[key] == dict_b[key]).all():
                return False
        else:
            if abs(dict_a[key] - dict_b[key]) > precision:
                return False
            if math.isnan(dict_a[key]) and not math.isnan(dict_b[key]):
                return False

    for key in dict_b:
        if key not in dict_a:
            return False
        if isinstance(dict_a[key], ndarray) and isinstance(dict_b[key], ndarray):
            if not (dict_a[key] == dict_b[key]).all():
                return False
        else:
            if abs(dict_a[key] - dict_b[key]) > precision:
                return False
            if math.isnan(dict_a[key]) and not math.isnan(dict_b[key]):
                return False

    return True
