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
from torch.utils.checkpoint import checkpoint

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


class SoftmaxAttention(nn.Module):
    '''
    Vanilla softmax attention
    '''
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

    def extra_repr(self) -> str:
        '''
        Empty meta-info of this kernel
        '''
        return "_"

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, V):
        return V

class Attention(nn.Module):
    '''
    General attention with K/Q/V projection
    '''
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]
        self.dim = config["embedding_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        if self.attn_type == "performer-256":
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        if self.attn_type == "mimoformer":
            from attention_mimoformer import MimoformerAttention
            self.attn = MimoformerAttention(config)
        if self.attn_type == "blockformer":
            from attention_blockformer import BlockformerAttention
            self.attn = BlockformerAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
            else:
                attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out
    
    def extra_repr(self) -> str:
        return self.attn.extra_repr()

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
