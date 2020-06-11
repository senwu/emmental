"""Emmental RNN module unit tests."""
import logging

import torch

from emmental.modules.rnn_module import RNN
from emmental.utils.utils import pad_batch

logger = logging.getLogger(__name__)


def test_rnn_module(caplog):
    """Unit test of RNN Module."""
    caplog.set_level(logging.INFO)

    n_class = 2
    emb_size = 10
    lstm_hidden = 20
    batch_size = 3
    seq_len = 4

    # Single direction RNN
    rnn = RNN(
        num_classes=n_class,
        emb_size=emb_size,
        lstm_hidden=lstm_hidden,
        attention=True,
        dropout=0.2,
        bidirectional=False,
    )
    _, input_mask = pad_batch(torch.randn(batch_size, seq_len))

    assert rnn(torch.randn(batch_size, seq_len, emb_size)).size() == (3, n_class)
    assert rnn(torch.randn(batch_size, seq_len, emb_size), input_mask).size() == (
        3,
        n_class,
    )

    # Bi-direction RNN
    rnn = RNN(
        num_classes=0,
        emb_size=emb_size,
        lstm_hidden=lstm_hidden,
        attention=False,
        dropout=0.2,
        bidirectional=True,
    )

    _, input_mask = pad_batch(torch.randn(batch_size, seq_len))

    assert rnn(torch.randn(batch_size, seq_len, emb_size)).size() == (
        3,
        2 * lstm_hidden,
    )
    assert rnn(torch.randn(batch_size, seq_len, emb_size), input_mask).size() == (
        3,
        2 * lstm_hidden,
    )
