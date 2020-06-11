"""Emmental identity module unit tests."""
import logging

import torch

from emmental.modules.identity_module import IdentityModule

logger = logging.getLogger(__name__)


def test_identity_module(caplog):
    """Unit test of Identity Module."""
    caplog.set_level(logging.INFO)

    identity_module = IdentityModule()

    input = torch.randn(10, 10)
    assert torch.equal(input, identity_module(input))
