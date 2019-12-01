import logging

import numpy as np
import torch

from emmental.modules.sparse_linear_module import SparseLinear

logger = logging.getLogger(__name__)


def test_sparse_linear_module(caplog):
    """Unit test of Sparse Linear Module"""

    caplog.set_level(logging.INFO)

    # Test w/o bias
    sparse_linear_module = SparseLinear(10, 2)

    index = torch.from_numpy(np.random.randint(10, size=100)).view(10, 10)
    weight = torch.randn(10, 10)

    assert isinstance(sparse_linear_module(index, weight), torch.Tensor)
    assert sparse_linear_module(index, weight).size() == (10, 2)

    # Test with bias
    sparse_linear_module = SparseLinear(10, 2, bias=True)

    assert isinstance(sparse_linear_module(index, weight), torch.Tensor)
    assert sparse_linear_module(index, weight).size() == (10, 2)
