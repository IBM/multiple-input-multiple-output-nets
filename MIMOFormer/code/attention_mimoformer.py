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

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class MimoformerAttention(FastAttention):
    '''
    MIMOFormer attention (att.)
    ''' 
    def __init__(self, config):

        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]

        kernel_fn = nn.ReLU() if self.kernel_type =="relu" else torch.exp
        super().__init__(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = kernel_fn)

        self.forward = self.forward_mimoformer

        self.mimo_M = config["mimo_M"]
        self.mimo_N = config["mimo_N"]
        bindings = 2 * torch.bernoulli(torch.ones((1, self.mimo_M, self.mimo_N,1,1, self.head_dim))/2) - 1
        self.register_buffer('bindings',bindings)

    def forward_mimoformer(self, Q, K, V, mask):
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

        # extract necesary binding
        bindings = self.bindings[:,:self.mimo_M, :self.mimo_N]  

        # Reshape b -> b', M, N
        b_prime = int(B/(self.mimo_M*self.mimo_N))
        q = q.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)
        k = k.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)
        v = v.reshape(b_prime, self.mimo_M, self.mimo_N, H, L, D)

        # binding and bundling
        bound_queries = q * bindings # 
        bundled_queries = bound_queries.sum(axis=1) # [B', N, L, H, D]

        bound_keys = k * bindings # 
        bundled_keys = bound_keys.sum(axis=2) # [B',M, L, H, D]

        bound_values = v * bindings # 
        bundled_values = bound_values.sum(axis=2) # [B, M, H, L, D]

        # FAVOR+ kernel
        create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn,
                               projection_matrix = self.projection_matrix, device = device)
        
        # pass keys and queries through projection 
        queries_prime, keys_prime = map(create_kernel, (bundled_queries, bundled_keys))

        # compute A matrix
        A_matrix = torch.einsum('...mhld,...mhlr->...hrd', (bundled_values, keys_prime)) # [B, H, R, D]

        # compute numerator
        numerator = torch.einsum('...hrd, ...nhlr -> ...nhld', (A_matrix, queries_prime)).unsqueeze(1) # [B, 1, N, H, L, D]

        # B matrix and denominator 
        B_matrix = torch.sum(keys_prime, dim=3) # [B, M, H, R]
        denominator = torch.einsum('...mhr, ...nhlr -> ...mnhl', (B_matrix, queries_prime)).unsqueeze(-1) # [B, M, N, H, L, 1]

        result = numerator / denominator # [B, M, N, H, L, D] (where numerator part is a superposition over 1...M also with noise terms with denominator already incorporated)

        # final unbinding to retriev ALL elements from the superposition
        result = result * bindings

        return result.reshape(B, H, L, D)

    def extra_repr(self):
        '''
        Skip connection: superpose M key-value pairs
        '''
        return f"rp_dim={self.rp_dim}_kernel_type={self.kernel_type}_M={self.mimo_M}_N={self.mimo_N}" 
    

    def set_mimo_MN(self, M, N):
        '''
        
        ''' 
        self.mimo_M = M
        self.mimo_N = N

# non-causal linear attention
def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), 
                       kernel_epsilon = 0.001, normalize_data = True, device = None):

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    # Actual projection
    projection = projection_matrix.type_as(data)
    data_dash = F.linear((data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)