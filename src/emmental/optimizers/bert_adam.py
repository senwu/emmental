# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team
# and Sen Wu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import logging
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class BertAdam(Optimizer):
    r"""Implements BERT version of Adam algorithm with weight decay fix.

    Args:
      params(iterable): Iterable of parameters to optimize or dicts defining
        parameter groups.
      lr(float, optional): Learning rate, defaults to 1e-3.
      betas(Tuple[float, float], optional): Coefficients used for computing running
        averages of gradient and its square, defaults to (0.9, 0.999).
      eps(float, optional): Term added to the denominator to improve numerical
        stability, defaults to 1e-6.
      weight_decay(float, optional): Weight decay (L2 penalty), defaults to 0.01.

    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)  # type: ignore

    def step(self, closure: Optional[Callable] = None) -> Any:
        """Performs a single optimization step.

        Args:
          closure(callable, optional): A closure that reevaluates the model and returns
            the loss, defaults to None.

        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:  # type: ignore
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider"
                        "SparseAdam instead"
                    )

                state = self.state[p]  # type: ignore

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = exp_avg / (exp_avg_sq.sqrt() + group["eps"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                p.data.add_(-group["lr"] * update)

                state["step"] += 1

        return loss
