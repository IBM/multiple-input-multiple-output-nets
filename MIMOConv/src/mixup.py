#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import numpy as np
import torch

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

# Mixup augmentation strategy: Beyond Empirical Risk Minimization https://arxiv.org/abs/1710.09412 
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    r"""Applies Mixup Augmentation to Data
    Args:
        x: input data
        y: labels
        alpha: high alpha -> even mix, low alpha -> randomly favors either input or permuted input
        use_cuda: whether to use cuda
    Returns:
        mixed input, labels and permuted labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    r"""evaluates a loss function/criterion for mixup data
    Args:
        criterion: criterion/loss function
        pred: predicted labels
        y_a: ground truth for first input mixed
        y_b: ground truth for second input mixed
        lam: lambda, i.e. how much weight to put on which input in mix
    Returns:
        weighted criterion score according to weight in mixed input lambda
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)