#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from typing import List
from models.superwideresnet import SuperWideResnet, BasicBlock
from superposition import noOrthoRegularization
from typing import Optional

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
class SReLU(nn.Module):
    def __init__(self, nc, relu_parameter = -1):
        r"""initialises shifted ReLU
        Args:
            nc: number of image channels of input/output tensor
            relu_parameter: initial value of the offset
        """
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, relu_parameter)

    def forward(self, x):
        r"""applies sReLU
        Args:
            x: input tensor
        Returns:
            output tensor after application of sReLU
        """
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias

class BasicISOBlock(BasicBlock):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, norm_layer: Optional[nn.Module] = None, relu_parameter: float = -1, skip_init: bool = False) -> None:
        r"""Coincides with BasicBlock (configured as in wideresnet), but with shifted relu. Initializes a block of a resnet model
        Args:
            inplanes: number of channels of input tensor
            planes: number of channels of output tensor and intermediate tensor after first convolution
            stride: stride to apply in first convolution
            downsample: None or 1x1 convolution in case direct skip connection is not possible (channel difference)
            norm_layer: normalization layer like BatchNorm
            relu_parameter: initialisation of sReLU parameter
            skip_init: whether skipInit is used
        """
        super().__init__(inplanes, planes, stride, downsample, norm_layer)
        if relu_parameter is None:
            relu_parameter = -1

        self.relu1 = SReLU(inplanes, relu_parameter=relu_parameter)
        self.relu2 = SReLU(planes, relu_parameter=relu_parameter)
        self.alpha = torch.nn.parameter.Parameter(data=torch.tensor(0.), requires_grad=True)
        self.skip_init = skip_init

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # order of operations as described in wideresnet paper which is different to the original resnet implementation
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.skip_init:
            out = self.alpha*out + identity
        else:
            out = out + identity

        return out

class AdjustedISOBlock(BasicBlock):

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, norm_layer: Optional[nn.Module] = None, relu_parameter: float = 0.5, skip_init: bool = False) -> None:
        r"""Coincides with BasicBlock (configured as in wideresnet), but with parametric relu. Initializes a block of a resnet model
        Args:
            inplanes: number of channels of input tensor
            planes: number of channels of output tensor and intermediate tensor after first convolution
            stride: stride to apply in first convolution
            downsample: None or 1x1 convolution in case direct skip connection is not possible (channel difference)
            norm_layer: normalization layer like BatchNorm
            relu_parameter: initialisation of pReLU parameter
            skip_init: whether skipInit is used
        """
        super().__init__(inplanes, planes, stride, downsample, norm_layer)
        if relu_parameter is None:
            relu_parameter = 0.5

        self.relu1 = nn.PReLU(num_parameters=inplanes, init=relu_parameter)
        self.relu2 = nn.PReLU(num_parameters=planes, init=relu_parameter)
        self.alpha = torch.nn.parameter.Parameter(data=torch.tensor(0.), requires_grad=True)
        self.skip_init = skip_init
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # order of operations as described in wideresnet paper which is different to the original resnet implementation
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.skip_init:
            out = self.alpha*out + identity
        else:
            out = out + identity

        return out

class SuperWideISONet(SuperWideResnet):
    def __init__(self, num_img_sup_cap: int, binding_type: str, width: int, layers: List[int], initial_width: int = None, num_classes: int = 1000, norm_layer:nn.Module = None, block=BasicISOBlock, dirac_init=False, relu_parameter=None, skip_init: bool = False, trainable_keys = True, input_channels: int=3) -> None:
        r"""Superposition based ISONet, inheriting from SuperWideResNet, but adjusted accordingly
        Args:
            num_img_sup_cap: num_img_sup_cap(acity) indicates how many images can at most be processed concurrently. Each batch is divided into num_img_sup parts, which are processed concurrently, with leftovers discarded.
            binding_type: One of HRR, MBAT or None
            width: width of WideResNet
            layers: list of layer sizes where each layer size indicates the number of blocks in a given layer
            initial_width: initial width of WideResNet, i.e. multiplies number of channels after first convolution.
            num_classes: number of classes/output logits
            norm_layer: which norm layer to use. If set to None, uses BatchNorm
            block: which ResNet block to use
            relu_parameter: initialization of pReLU/sReLU parameters in subclass superWideISONet
            skip_init: whether to use skipInit
            trainable_keys: whether to NOT freeze the binding keys
        """
        SuperWideResnet.__init__(self, num_img_sup_cap=num_img_sup_cap, binding_type=binding_type, width=width, layers=layers, initial_width=initial_width, num_classes=num_classes, norm_layer=norm_layer, block=block, relu_parameter=relu_parameter, skip_init=skip_init, trainable_keys = trainable_keys, input_channels = input_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, noOrthoRegularization):
                if isinstance(block, BasicISOBlock):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.45)
                elif isinstance(block, AdjustedISOBlock):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.5)

            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # dirac initialisation
        if dirac_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and not isinstance(m, noOrthoRegularization):
                    # for the first 7x7 / 3x3 convolution we use pytorch default initialization
                    # and do not enforce orthogonality since the large input/output channel difference
                    # also, binding and unbinding themselves are not orthogonalized
                    nn.init.dirac_(m.weight)

        
    def ortho_conv(self, m, device):
        r"""helper to enforces isometry
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
        gram = F.conv2d(operator, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1),
                        stride=m.stride, groups=1)
        identity = torch.zeros(size=gram.shape, device=device)
        identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(operator.shape[0], device=device, dtype=torch.float64)
        return torch.sum((gram - identity) ** 2.0) / 2.0

    def isometry_regularization(self, device):
        r"""enforces isometry for 2d-convolutional operation as described in ISONet paper https://arxiv.org/abs/2006.16992
        Args:
            device: current device used, i.e. GPU/CPU
        Returns:
            orthogonal convolution regularization term
        """
        ortho_penalty = torch.zeros(size=(1,), device=device, dtype=torch.float64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, noOrthoRegularization):
                ortho_penalty += self.ortho_conv(m, device)
        return ortho_penalty

    #------------- Inspect Training behaviour ---------------

    def compute_average_abs_alpha(self, device):
        abs_alpha_sum = torch.zeros(size=(1,), device=device)
        count = 0
        for mod in self.modules():
            if isinstance(mod, BasicBlock):
                abs_alpha_sum += torch.abs(mod.alpha.data)
                count += 1
        return 0 if count == 0 else abs_alpha_sum / count

    def compute_average_relu_param(self, device):
        # each layer is equally weighted in total average independent of width
        param_sum = torch.zeros(size=(1,), device=device)
        count = 0
        for mod in self.modules():
            if isinstance(mod, AdjustedISOBlock):
                param_sum += torch.mean(mod.relu1.weight)
                param_sum += torch.mean(mod.relu2.weight)
                count += 2
            elif isinstance(mod, BasicISOBlock):
                param_sum += torch.mean(mod.relu1.srelu_bias)
                param_sum += torch.mean(mod.relu1.srelu_bias)
                count += 2
        return 0 if count == 0 else param_sum / count

    def compute_relu_param_variance(self, device):
        # each layer is equally weighted in total average independent of width
        average = self.compute_average_relu_param(device)
        sum = torch.zeros(size=(1,), device=device)
        count = 0
        for mod in self.modules():
            if isinstance(mod, SReLU):
                sum += torch.mean((mod.srelu_bias.data - average)**2)
                count += 1
            if isinstance(mod, nn.PReLU):
                sum += torch.mean((mod.weight - average)**2)
                count += 1
        return 0 if count == 0 else sum.item() / count