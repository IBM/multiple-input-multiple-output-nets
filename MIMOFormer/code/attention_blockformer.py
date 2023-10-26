#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from performer_pytorch import FastAttention 
from functools import partial
from attention_mimoformer import generalized_kernel

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class BlockformerAttention(FastAttention):
    '''
    MIMOFormer (att.+MLP)

    This MIMOFormer attention is used in the MIMOFormer (att.+MLP) configuration. 
    In contrast to MIMOFormer (att.), this module yields an superposition output. 
    '''
    def __init__(self, config):
        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]

        kernel_fn = nn.ReLU() if self.kernel_type =="relu" else torch.exp
        super().__init__(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = kernel_fn)

        self.mimo_M = config["mimo_M"]
        self.mimo_N = config["mimo_N"]
        bindings = 2 * torch.bernoulli(torch.ones((1, self.mimo_M, self.mimo_N,1,1, self.head_dim))/2) - 1
        bindings_embedding = 2 * torch.bernoulli(torch.ones((1, self.mimo_M, self.mimo_N, 1,config["embedding_dim"]))/2) - 1
        self.register_buffer('bindings',bindings)
        self.register_buffer('bindings_embedding',bindings_embedding)

    def forward(self, Q, K, V, mask):
        '''
        FAVOR+S attention
        '''
        # Rescaling query and keys
        scale = math.sqrt(math.sqrt(self.head_dim)*self.mimo_M*self.mimo_N)
        q = Q / scale        
        k = K * mask[:, None, :, None] / scale
        v = V * mask[:, None, :, None]

        device = q.device
        B, H, L, D = q.shape

        # reshape b -> b', M, N
        b_prime = int(B/(self.mimo_M*self.mimo_N))
        q = q.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)
        k = k.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)
        v = v.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)

        # binding and bundling
        bound_queries = q * self.bindings # [B', M, N, H, L, D]
        bundled_queries = bound_queries.sum(axis=1) # [B', N, L, H, D]

        bound_keys = k * self.bindings # [B', M, N, H, L, D]
        bundled_keys = bound_keys.sum(axis=2) # [B',M, L, H, D]

        bound_values = v * self.bindings # [B',M, N, H, L, D]
        bundled_values = bound_values.sum(axis=2) # [B, M, H, L, D]
        
        # FAVOR+ kernel
        create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn,
                               projection_matrix = self.projection_matrix, device = device)

        # pass keys and queries through projection 
        queries_prime, keys_prime = map(create_kernel, (bundled_queries, bundled_keys))
        
        # compute A matrix
        A_matrix = torch.einsum('...mhld,...mhlr->...hrd', (bundled_values, keys_prime)) # [B, H, R, D]

        # compute numerator
        numerator = torch.einsum('...hrd, ...nhlr -> ...nhld', (A_matrix, queries_prime)) # [B, N, H, L, D]

        # B matrix and denominator 
        B_matrix = torch.sum(keys_prime, dim=(1,3)) # [B, H, R]
        denominator = torch.einsum('...hr, ...nhlr -> ...nhl', (B_matrix, queries_prime)).unsqueeze(-1) # [B, M, N, H, L, 1]

        # final result of FAVOR+S attention (still a superposition)
        result = numerator / denominator # [B, N, H, L, D] 

        return result.reshape(b_prime*self.mimo_N, H, L, D) 
    
    def retrieve(self, X):
        '''
        Retrieve M elements by unbinding 
        '''
        bs, L, d = X.shape 
        X = X.reshape(-1,1, self.mimo_N,L,d)*self.bindings_embedding
        return X.reshape(bs*self.mimo_M,L, d)

    def skipconnection(self, X):
        '''
        Skip connection: superpose M key-value pairs
        '''
        bs, L, d = X.shape 
        bs_prime = int(bs/(self.mimo_M*self.mimo_N))
        bound_X = X.reshape(bs_prime, self.mimo_M, self.mimo_N, L, d)*self.bindings_embedding
        bundled_X = bound_X.sum(axis=1) # [B', N, L, H, D]
        # flatten out mimo_N dimension here
        return bundled_X.reshape(bs_prime*self.mimo_N, L, d)

    def extra_repr(self):
        '''
        Just for retrieving meta-info of this kernel
        '''
        return f"rp_dim={self.rp_dim}_kernel_type={self.kernel_type}_M={self.mimo_M}_N={self.mimo_N}" 
