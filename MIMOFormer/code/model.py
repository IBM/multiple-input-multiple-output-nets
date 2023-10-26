#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
import torch.nn as nn
import numpy as np
import math
from attention import Attention

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config["embedding_dim"]
        self.learn_pos_emb = config["learn_pos_emb"]
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(num_embeddings = config["vocab_size"], embedding_dim = config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, mean = 0.0, std = 0.02) 

        # Position embeddings
        if self.learn_pos_emb:
            self.position_embeddings = nn.Embedding(num_embeddings = config["max_seq_len"], embedding_dim = config["embedding_dim"])
            torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()
        
        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        if self.learn_pos_emb:
            X_pos = self.position_embeddings(position_ids)
        else:
            X_pos = self.fixed_pos_emb(seq_len = seq_len, device = input_ids.device).view(1, seq_len, self.dim).expand(batch_size, -1, -1)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.act = nn.ReLU 
        self.norm1 = nn.LayerNorm(config["embedding_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["embedding_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["transformer_hidden_dim"]),
            self.act(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["embedding_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X
    
    def extra_repr(self): 
        return self.mha.extra_repr()

    def ortho_FCL(self, m, device):
        """helper to enforces isometry
        Args:
            m: module
            device: current device used, i.e. GPU/CPU
        Returns:
            orthogonal convolution regularization term for module
        """
        operator = m.weight
        transposed = operator.shape[1] < operator.shape[0]
        if transposed:
            operator = operator.transpose(1, 0)
        gram = operator@operator.T
        identity = torch.eye(operator.shape[0], device=device, dtype=torch.float64)
        return torch.sum((gram - identity) ** 2.0) / 2.0

    def isometry_regularization(self, device): 
        """
        enforces isometry for FCL operation as described in ISONet paper https://arxiv.org/abs/2006.16992
        Args:
            device: current device used, i.e. GPU/CPU
        Returns:
            orthogonal FCL regularization term
        """
        ortho_penalty = torch.zeros(size=(1,), device=device, dtype=torch.float64)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                ortho_penalty += self.ortho_FCL(m, device)
        return ortho_penalty

class BLOCKFormer(Transformer):
    '''
    MIMOFormer (att.+MLP)
    Implements top level layer, which is needed for skipconnection reductino and retrieval. 
    '''
    def __init__(self, config):
        super().__init__(config)

    def forward(self, X, mask):
        X = self.dropout1(self.mha(self.norm1(X), mask))  + self.mha.attn.skipconnection(X)
        X = self.mlpblock(self.norm2(X)) + X
        X = self.mha.attn.retrieve(X)
        return X
    


class Model(nn.Module):
    '''
    Implements embeddings + num_layers * Xformer layers
    '''
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        xformer = BLOCKFormer if config["attn_type"] =="blockformer" else Transformer 

        if self.tied_weights:
            self.transformer = xformer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", xformer(config))
        
        self.norm = nn.LayerNorm(config["embedding_dim"])

    def forward(self, input_ids, mask = None):
        
        input_ids = input_ids.type(torch.int)
        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
    
    def extra_repr(self): 
        # Return attention meta data
        if self.tied_weights:
            return self.transformer.extra_repr()
        else:
            return self.transformer_0.extra_repr()

    def isometry_regularization(self, device): 
        """
        Enforces isometry for FCL operation as described in ISONet paper https://arxiv.org/abs/2006.16992
        Args:
            device: current device used, i.e. GPU/CPU
        Returns:
            orthogonal FCL regularization term
        """
        ortho_penalty = torch.zeros(size=(1,), device=device, dtype=torch.float64)

        if self.tied_weights:
            for idx in range(self.num_layers):
                ortho_penalty += self.transformer.isometry_regularization(device)
        else:
            for idx in range(self.num_layers):
                ortho_penalty += getattr(self, f"transformer_{idx}").isometry_regularization(device)
        return ortho_penalty
    
    def start_MIMO_warmup(self): 
        '''
        This is only for MIMOFormer (att.)
        Set N and M to N'=N/2 and M'=M/2
        '''
        if self.tied_weights:
            for idx in range(self.num_layers):
                obj = self.transformer.mha.attn
                # intermediate storage or original M, N
                obj.orig_mimoM = obj.mimo_M
                obj.orig_mimoN = obj.mimo_N
                # set new M, N
                obj.set_mimo_MN(obj.mimo_M//2,obj.orig_mimoN//2)
        else:
            for idx in range(self.num_layers):
                obj = getattr(self, f"transformer_{idx}").mha.attn
                # intermediate storage or original M, N
                obj.orig_mimoM = obj.mimo_M
                obj.orig_mimoN = obj.mimo_N
                # set new M, N
                obj.set_mimo_MN(obj.mimo_M//2,obj.orig_mimoN//2)
            

    def stop_MIMO_warmup(self): 
        '''
        This is only for MIMOFormer (att.)
        Set N and M to original values
        '''
        if self.tied_weights:
            for idx in range(self.num_layers):
                obj = self.transformer.mha.attn
                obj.set_mimo_MN(obj.orig_mimoM,obj.orig_mimoN)
        else:
            for idx in range(self.num_layers):
                obj = getattr(self, f"transformer_{idx}").mha.attn
                obj.set_mimo_MN(obj.orig_mimoM,obj.orig_mimoN)