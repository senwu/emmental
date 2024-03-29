"""Emmental utils unit tests."""
import logging
from functools import partial

import numpy as np
import pytest
import torch

from emmental.utils.utils import (
    array_to_numpy,
    construct_identifier,
    convert_to_serializable_json,
    merge,
    move_to_device,
    nullable_float,
    nullable_int,
    nullable_string,
    pred_to_prob,
    prob_to_pred,
    random_string,
    str2bool,
    str2dict,
    str2list,
)


def test_prob_to_pred(caplog):
    """Unit test of prob_to_pred."""
    caplog.set_level(logging.INFO)

    assert (
        np.array_equal(prob_to_pred(np.array([[0, 1], [1, 0]])), np.array([1, 0]))
        is True
    )
    assert (
        np.array_equal(
            prob_to_pred(np.array([[0.4, 0.5], [0.2, 0.8], [0.9, 0.1]])),
            np.array([1, 1, 0]),
        )
        is True
    )

    with pytest.raises(ValueError):
        prob_to_pred(1.23)


def test_pred_to_prob(caplog):
    """Unit test of pred_to_prob."""
    caplog.set_level(logging.INFO)

    assert np.array_equal(
        pred_to_prob(np.array([0, 1, 2]), 3),
        np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
    )
    assert np.array_equal(
        pred_to_prob(np.array([0, 1, 2]), 4),
        np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]),
    )


def test_move_to_device(caplog):
    """Unit test of move_to_device."""
    caplog.set_level(logging.INFO)

    assert torch.equal(move_to_device(torch.Tensor([1, 2]), -1), torch.Tensor([1, 2]))
    assert move_to_device({1: torch.tensor([1, 2]), 2: torch.tensor([3, 4])}, -1)
    assert move_to_device([torch.tensor([1, 2]), torch.tensor([3, 4])], -1)
    assert move_to_device((torch.tensor([1, 2]), torch.tensor([3, 4])), -1)


def test_array_to_numpy(caplog):
    """Unit test of array_to_numpy."""
    caplog.set_level(logging.INFO)

    assert (
        np.array_equal(array_to_numpy([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))
        is True
    )
    assert (
        np.array_equal(
            array_to_numpy(torch.tensor([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]])
        )
        is True
    )
    assert np.array_equal(
        array_to_numpy([[1, 2], [3, 4]], flatten=True), np.array([1, 2, 3, 4])
    )

    with pytest.raises(ValueError):
        array_to_numpy(1.23)


def test_merge(caplog):
    """Unit test of merge."""
    caplog.set_level(logging.INFO)

    assert merge({1: 1}, {2: 2}) == {1: 1, 2: 2}
    assert merge({1: 1}, None) == {1: 1}
    assert merge(None, {2: 2}) == {2: 2}
    assert merge({1: 1, 3: {4: 4}}, {2: 2}) == {1: 1, 2: 2, 3: {4: 4}}


def test_str2bool(caplog):
    """Unit test of str2bool."""
    caplog.set_level(logging.INFO)

    assert str2bool("Yes") is True
    assert str2bool("YeS") is True
    assert str2bool("TRUE") is True
    assert str2bool("True") is True
    assert str2bool("T") is True
    assert str2bool("Y") is True
    assert str2bool("1") is True

    assert str2bool("N") is False
    assert str2bool("No") is False
    assert str2bool("False") is False
    assert str2bool("n") is False
    assert str2bool("f") is False
    assert str2bool("0") is False

    with pytest.raises(ValueError):
        str2bool("o")


def test_str2dict(caplog):
    """Unit test of str2dict."""
    caplog.set_level(logging.INFO)

    assert str2dict("1:1") == {"1": "1"}
    assert str2dict("1:1,2:2") == {"1": "1", "2": "2"}


def test_str2list(caplog):
    """Unit test of str2list."""
    caplog.set_level(logging.INFO)

    assert str2list("1,2,3") == ["1", "2", "3"]
    assert str2list("1,2:3", ":") == ["1,2", "3"]


def test_nullable_float(caplog):
    """Unit test of nullable_float."""
    caplog.set_level(logging.INFO)

    assert nullable_float("none") is None
    assert nullable_float("1.2") == 1.2


def test_nullable_int(caplog):
    """Unit test of nullable_int."""
    caplog.set_level(logging.INFO)

    assert nullable_int("none") is None
    assert nullable_int("1") == 1


def test_nullable_string(caplog):
    """Unit test of nullable_string."""
    caplog.set_level(logging.INFO)

    assert nullable_string("none") is None
    assert nullable_string("1") == "1"


def test_construct_identifier(caplog):
    """Unit test of construct_identifier."""
    caplog.set_level(logging.INFO)

    assert construct_identifier("1", "2", "3", "4") == "1/2/3/4"
    assert construct_identifier("1", "2", "3") == "1/2/3"


def test_random_string(caplog):
    """Unit test of random_string."""
    caplog.set_level(logging.INFO)

    assert len(random_string(10)) == 10
    assert len(random_string(5)) == 5
    assert random_string(5).islower() is True


def test_convert_to_serializable_json(caplog):
    """Unite test of convert_to_serializable_json."""
    caplog.set_level(logging.INFO)

    class ABC:
        a = 1

    def cde(a, b):
        return a, b

    config = {
        1: 1,
        2: 2,
        3: {4: cde, 5: [ABC(), {1: 1, 2: partial(cde, 1)}]},
        6: (ABC(), 1),
    }

    assert convert_to_serializable_json(config) == {
        1: 1,
        2: 2,
        3: {4: "Function: cde", 5: ["Class: ABC", {1: 1, 2: "Function: cde"}]},
        6: ("Class: ABC", 1),
    }
