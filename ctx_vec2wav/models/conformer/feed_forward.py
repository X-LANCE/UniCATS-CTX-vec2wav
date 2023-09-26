# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

import torch.nn as nn
from torch import Tensor

from ctx_vec2wav.models.conformer.activation import Swish
from ctx_vec2wav.models.conformer.modules import Linear
from ctx_vec2wav.models.conformer.espnet_conv_ffn import MultiLayeredConv1d


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    to regularize the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
            ffn_type: str = 'linear',
            conv_kernel_size: int = None
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.ffn_type = ffn_type
        if ffn_type == 'linear':
            self.sequential = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                Linear(encoder_dim, int(encoder_dim * expansion_factor), bias=True),
                Swish(),
                nn.Dropout(p=dropout_p),
                Linear(int(encoder_dim * expansion_factor), encoder_dim, bias=True),
                nn.Dropout(p=dropout_p),
            )
        elif ffn_type == 'conv1d':
            self.sequential = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                MultiLayeredConv1d(in_chans=encoder_dim, hidden_chans=int(encoder_dim * expansion_factor),
                                                 kernel_size=conv_kernel_size, dropout_rate=dropout_p)
            )
        else:
            raise NotImplementedError

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
