"""Emmental list to tensor unit tests."""
import logging

from emmental.utils.seed import set_random_seed


def test_set_random_seed(caplog):
    """Unit test of setting random seed."""
    caplog.set_level(logging.INFO)

    set_random_seed(1)
    set_random_seed()
    set_random_seed(-999999999999)
    set_random_seed(999999999999)
