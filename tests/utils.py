import math


def isequal(dict_a, dict_b, precision=1e-10):
    for key in dict_a:
        if (
            key not in dict_b
            or abs(dict_a[key] - dict_b[key]) > precision
            or (math.isnan(dict_a[key]) and not math.isnan(dict_b[key]))
        ):
            return False

    for key in dict_b:
        if (
            key not in dict_a
            or abs(dict_a[key] - dict_b[key]) > precision
            or (math.isnan(dict_b[key]) and not math.isnan(dict_a[key]))
        ):
            return False

    return True
