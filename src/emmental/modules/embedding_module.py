from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor


class EmbeddingModule(nn.Module):
    r"""Embedding module.

    Args:
      word_counter(dict): Word count dictionary that contians the frequencies of
        each word, defaults to None.
      max_size(int): Max size of word dictionary, defaults to None.
      word_dim(int): Dimension of embeddings, defaults to 300.
      specials(list): The list of special tokens (e.g., padding or eos) that will
        be prepended to the vocabulary, defaults to [].
      threshold(int): The minimum frequency needed to include a token in the
        vocabulary, defaults to None.
      embedding_file(str): The pretrained embedding file path, defaults to None.
      fix_emb(bool): Whether fix word embeddings or not, defaults to False.

    """

    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(
        self,
        word_counter: Optional[Dict[str, int]] = None,
        max_size: Optional[int] = None,
        word_dim: int = 300,
        specials: List[str] = [],
        threshold: int = 0,
        embedding_file: Optional[str] = None,
        fix_emb: bool = False,
    ) -> None:

        super().__init__()
        assert (
            word_counter is not None or embedding_file is not None
        ), "word_counter and embedding_file are not provided."

        self.word_counter = word_counter
        self.dim = word_dim

        # remove words that occur less than threshold
        if self.word_counter and threshold > 1:
            self.word_counter = dict(
                [(k, v) for k, v in self.word_counter.items() if v >= threshold]
            )

        max_size = None if max_size is None else max_size + len(specials)
        reverse = True

        if embedding_file:
            emb_dim, emb_w2i, emb_wv = self._load_embedding(embedding_file)
            self.dim = emb_dim
            if word_counter is None:
                self.word_counter = emb_w2i
                reverse = False

        self.id2word = sorted(
            self.word_counter, key=lambda k: self.word_counter[k], reverse=reverse
        )

        specials = [self.UNK, self.PAD] + [
            special for special in specials if special not in [self.UNK, self.PAD]
        ]
        # append special tokens and remove duplicate words
        self.id2word = specials + [
            word for word in self.id2word if word not in specials
        ]

        # limit the word list size
        if max_size:
            self.id2word = self.id2word[:max_size]

        self.word2id = dict(
            [(self.id2word[idx], idx) for idx in range(len(self.id2word))]
        )
        self.size = len(self.id2word)

        # Initalize word embeddings
        self.embeddings = nn.Embedding(self.size, self.dim)
        self.embeddings.weight.data.uniform_(-1, 1)
        self.embeddings.weight.data[self.word2id[self.PAD]] = 0.0

        # Update word embedding with pretrained embeddings
        if embedding_file:
            for w, i in emb_w2i.items():
                if w in self.word2id:
                    self.word2id[w]
                    self.embeddings.weight.data[self.word2id[w]].copy_(  # type: ignore
                        torch.from_numpy(emb_wv[emb_w2i[w]])
                    )

        if fix_emb:
            self.embeddings.weight.requires_grad = False

    def _load_embedding(
        self, embedding_file: str
    ) -> Tuple[int, Dict[str, int], List[ndarray]]:
        r"""Load the pre-trained embeddings from file.

        Args:
          embedding_file: The pretrained embedding file path.

        Returns:
          tuple: word embedding dimension, word to index dict, and embedding vectors.

        """

        emb_dim = 0
        emb_w2i: Dict[str, int] = {}
        emb_wv = []

        with open(embedding_file, encoding="utf8") as f:
            for line in f:
                elems = line.split()

                # skip the header
                if len(elems) == 2:
                    continue

                # collect embedding dim
                emb_dim = len(elems) - 1
                # collect word
                token = elems[0]
                # collect word embedding
                if token not in emb_w2i:
                    emb_w2i[token] = len(emb_w2i)
                    emb_wv.append(np.array([float(v) for v in elems[1:]]))
        return emb_dim, emb_w2i, emb_wv

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        r"""Forward function.

        Args:
          input(Tensor): Input tensor.

        Returns:
          Tensor: Output tensor.

        """
        return self.embeddings(input)
