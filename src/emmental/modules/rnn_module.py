from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RNN(nn.Module):
    r"""A recurrent neural network module.

    Args:
      num_classes(int): Number of classes.
      emb_size(int): Dimension of embeddings.
      lstm_hidden(int): Size of LSTM hidden layer size.
      num_layers(int): Number of recurrent layers, defaults to 1.
      dropout(float): Dropout parameter of LSTM, defaults to 0.0.
      attention(bool): Use attention or not, defaults to True.
      bidirectional(bool): Use bidirectional LSTM or not, defaults to True.

    """

    def __init__(
        self,
        num_classes: int,
        emb_size: int,
        lstm_hidden: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention: bool = True,
        bidirectional: bool = True,
    ) -> None:

        super().__init__()

        self.emb_size = emb_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_layers = num_layers
        self.dropout = dropout
        # Number of dimensions of output. 0 means no final linear layer.
        self.num_classes = num_classes

        if self.num_classes > 0:
            self.final_linear = True
        else:
            self.final_linear = False

        self.drop = nn.Dropout(dropout)

        b = 2 if self.bidirectional else 1

        self.word_lstm = nn.LSTM(
            self.emb_size,
            self.lstm_hidden,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        if attention:
            self.attn_linear_w_1 = nn.Linear(
                b * lstm_hidden, b * lstm_hidden, bias=True
            )
            self.attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)

        if self.final_linear:
            self.linear = nn.Linear(b * lstm_hidden, num_classes)

    def forward(  # type: ignore
        self, x: Tensor, x_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Forward function.

        Args:
          x(Tensor): Input tensor.
          x_mask(Tensor, optional): Input mask tensor, defaults to None.

        Returns:
          Tensor: Output tensor.

        """
        x_emb = self.drop(x)
        output_word, state_word = self.word_lstm(x_emb)
        output_word = self.drop(output_word)
        if self.attention:
            """
            An attention layer where the attention weight is
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
            """
            word_squish = torch.tanh(self.attn_linear_w_1(output_word))
            word_attn = self.attn_linear_w_2(word_squish)
            if x_mask is not None:
                word_attn.data.masked_fill_(x_mask.data.unsqueeze(dim=2), float("-inf"))
            word_attn_norm = torch.sigmoid(word_attn.squeeze(2))
            word_attn_vectors = torch.bmm(
                output_word.transpose(1, 2), word_attn_norm.unsqueeze(2)
            ).squeeze(2)
            output = (
                self.linear(word_attn_vectors)
                if self.final_linear
                else word_attn_vectors
            )
        else:
            """
            Mean pooling
            """
            if x_mask is None:
                x_mask = x.new_full(x.size()[:2], fill_value=0, dtype=torch.bool)
            x_lens = x_mask.data.eq(0).long().sum(dim=1)
            weights = (
                output_word.new_ones(output_word.size())
                / x_lens.view(x_lens.size()[0], 1, 1).float()
            )
            weights.data.masked_fill_(x_mask.data.unsqueeze(dim=2), 0.0)
            word_vectors = torch.sum(output_word * weights, dim=1)
            output = self.linear(word_vectors) if self.final_linear else word_vectors

        return output
