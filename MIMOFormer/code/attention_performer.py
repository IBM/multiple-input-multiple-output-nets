#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
import torch.nn as nn
import math
from performer_pytorch import FastAttention

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class PerformerAttention(nn.Module):
    '''
    Performer (FAVOR+) attention
    '''
    def __init__(self, config):
        super().__init__()
        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]
        kernel_fn = nn.ReLU() if self.kernel_type == "relu" else torch.exp
        self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, 
                        generalized_attention=True, causal = False, kernel_fn = kernel_fn)

    def forward(self, Q, K, V, mask):
        '''
        FAVOR+ attention
        '''
        return self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)) * mask[:, None, :, None],
            V * mask[:, None, :, None])

    def extra_repr(self):
        '''
        Just for retrieving meta-info of this kernel
        '''
        return f'rp_dim={self.rp_dim}_kernel_type={self.kernel_type}'
