import numpy as np
import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):
    """An embedding module.

    :param word_counter: Word count dictionary that contians the frequencies of
        each word (default: None)
    :type word_counter: dict
    :param max_size: Max size of word dictionary (default: None)
    :type max_size: int
    :param word_dim: Dimension of embeddings (default: 300)
    :type word_dim: int
    :param specials: The list of special tokens (e.g., padding or eos) that will
        be prepended to the vocabulary. (default: [])
    :type specials: list
    :param threshold: The minimum frequency needed to include a token in the
        vocabulary (default: None)
    :type threshold: int
    :param embedding_file: The pretrained embedding file path (default: None)
    :type embedding_file: str
    :param fix_emb: Whether fix word embeddings or not (default: False)
    :type fix_emb: bool
    """

    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(
        self,
        word_counter=None,
        max_size=None,
        word_dim=300,
        specials=[],
        threshold=0,
        embedding_file=None,
        fix_emb=False,
    ):

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
                    self.embeddings.weight.data[self.word2id[w]].copy_(
                        torch.from_numpy(emb_wv[emb_w2i[w]])
                    )

        if fix_emb:
            self.embeddings.weight.requires_grad = False

    def _load_embedding(self, embedding_file):
        emb_dim = 0
        emb_w2i = {}
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

    def forward(self, input):
        return self.embeddings(input)
