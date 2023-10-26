#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
import torch.nn as nn
from torch import Tensor

# wrapper classes to prevent orthogonal regularization 
class noOrthoRegularization:
    pass
class noOrthoRegularizationConv1d(nn.Conv1d, noOrthoRegularization):
    pass
class noOrthoRegularizationConv2d(nn.Conv2d, noOrthoRegularization):
    pass

class Superposition:
    r"""implements binding and unbinding mechanisms such that subclasses can process superposed images. Both HRR and MBAT are listed"""

    def pointwiseConvThroughChannels(self, x: Tensor) -> Tensor:
        r"""implements convolution through channels, i.e. pixel-wise HRR binding
        Args:
            x: image tensor of size (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        Returns:
            image tensor of size (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        """
        x = x.permute(0,3,4,1,2) # yields (N/num_img_sup_cap, H, W, num_img_sup_cap, C)
        old_shape = x.shape
        x = x.reshape(x.shape[0]*x.shape[1]*x.shape[2], x.shape[3], x.shape[4]) # yields (N/num_img_sup_cap*H*W, num_img_sup_cap, C)
        x = self.channelConv(x)
        x = x[:, :, 1:] # circular padding is added on both sides, hence at the end (due to vectors being even), one channel was duplicated which has to be removed
        x = x.reshape(old_shape) # yields (N/num_img_sup_cap, H, W, num_img_sup_cap, C)
        x = x.permute(0,3,4,1,2) # yields (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        return x

    def conv1x1Stack(self, x:Tensor) -> Tensor:
        r"""implements pixel-wise MBAT binding
        Args:
            x: image tensor of size (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        Returns:
            image tensor of size (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        """
        old_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]) # (N/num_img_sup_cap, num_img_sup_cap * C, H, W), channels are stacked on top
        x = self.channelLinear(x)
        x = x.reshape(old_shape) # (N/num_img_sup_cap, num_img_sup_cap, C, H, W)
        return x

    def binding_regularization(self):
        r"""computes regularization term with the goal of orthonormalizing binding keys"""
        if self.binding_type == "HRR":
            weights = self.channelConv.weight.reshape(self.num_img_sup_cap, -1)
            normed_weights = nn.functional.normalize(weights, p=2, dim=1)
            # returns average squared cosine of angle between any two pairs of vectors + average squared error of norm to length 1
            avg_abs_inner_product = (torch.norm(torch.triu(torch.matmul(normed_weights, torch.transpose(normed_weights, 0, 1)), diagonal=1)))**2 / (self.num_img_sup_cap*(self.num_img_sup_cap-1)/2) if self.num_img_sup_cap > 1 else 0
            avg_norm_delta = (torch.norm(torch.norm(weights, dim=1) - 1))**2 / self.num_img_sup_cap
            return avg_abs_inner_product + avg_norm_delta
        elif self.binding_type == "MBAT":
            return 0
        else: # includes case "None"
            return 0

    def __init__(self, num_img_sup_cap:int, binding_type:str, channels:int, fully_connected_features:int, trainable_keys = True):
        r"""Initializes Superposition Capabilities
        Args:
            num_img_sup_cap: number of superposition channels present
            binding_type: either HRR, MBAT or None
            channels: number of channels in image tensor
            fully_connected_features: number of features/neurons before fully-connected layer is applied
            trainable_keys: whether keys are NOT frozen during training
        """
        assert channels % 2 == 0, "Implementation of convolution through channels can only handle even number of channels."

        if binding_type == "HRR":
            self.channelConv = noOrthoRegularizationConv1d(in_channels = num_img_sup_cap, out_channels = num_img_sup_cap, kernel_size = channels, padding = channels//2, padding_mode = 'circular', groups = num_img_sup_cap, bias = False)
            if not trainable_keys:
                self.channelConv.requires_grad_(False)
        if binding_type == "MBAT":
            self.channelLinear = noOrthoRegularizationConv2d(in_channels = channels*num_img_sup_cap, out_channels = channels*num_img_sup_cap, kernel_size = 1, groups = num_img_sup_cap, bias = False)
            if not trainable_keys:
                self.channelLinear.requires_grad_(False)

        assert num_img_sup_cap >= 1, "num_img_sup_cap is not a natural number"
        self.num_img_sup_cap = num_img_sup_cap

        assert binding_type in ["None", "HRR", "MBAT"], "binding_type is invalid"
        self.binding_type = binding_type

        # Set standard unbinding, used for HRR and MBAT (essentially a fully connected layer if run after pooling)
        self.unbind = nn.Sequential(noOrthoRegularizationConv2d(fully_connected_features, num_img_sup_cap*fully_connected_features, kernel_size=1, stride=1), 
                                    nn.ReLU(inplace=True))
                                    
        if binding_type == "None":
            self.bind = nn.Identity()
            if num_img_sup_cap == 1:
                self.unbind = nn.Identity()
        if binding_type == "HRR":
            self.bind = self.pointwiseConvThroughChannels
        elif binding_type == "MBAT":
            self.bind = self.conv1x1Stack
