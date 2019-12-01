import logging

import torch

from emmental.modules.embedding_module import EmbeddingModule
from emmental.utils.utils import set_random_seed

logger = logging.getLogger(__name__)


def test_embedding_module(caplog):
    """Unit test of Embedding Module"""

    caplog.set_level(logging.INFO)

    # Set random seed seed
    set_random_seed(1)

    word_counter = {"1": 1, "2": 3, "3": 1}
    weight_tensor = torch.FloatTensor(
        [
            [-0.4277, 0.7110, -0.3268, -0.7473, 0.3847],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2247, -0.7969, -0.4558, -0.3063, 0.4276],
            [2.0000, 2.0000, 2.0000, 2.0000, 2.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        ]
    )

    emb_layer = EmbeddingModule(word_counter=word_counter, word_dim=10, max_size=10)

    assert emb_layer.dim == 10
    # <unk> and <pad> are default tokens
    assert emb_layer.embeddings.weight.size() == (5, 10)

    emb_layer = EmbeddingModule(
        word_counter=word_counter,
        word_dim=10,
        embedding_file="tests/modules/embeddings.vec",
        fix_emb=True,
    )

    assert emb_layer.dim == 5
    assert emb_layer.embeddings.weight.size() == (5, 5)
    assert torch.max(torch.abs(emb_layer.embeddings.weight.data - weight_tensor)) < 1e-4

    assert (
        torch.max(
            torch.abs(emb_layer(torch.LongTensor([1, 2])) - weight_tensor[1:3, :])
        )
        < 1e-4
    )

    # With threshold
    word_counter = {"1": 3, "2": 1, "3": 1}
    emb_layer = EmbeddingModule(word_counter=word_counter, word_dim=10, threshold=2)
    assert emb_layer.embeddings.weight.size() == (3, 10)

    # No word counter
    emb_layer = EmbeddingModule(embedding_file="tests/modules/embeddings.vec")
    assert emb_layer.embeddings.weight.size() == (5, 5)
