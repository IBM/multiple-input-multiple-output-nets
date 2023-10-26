#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
from torch import Tensor, nn
from typing import List, Optional, Type
from superposition import Superposition
from superposition import noOrthoRegularizationConv2d

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
# Based on Wide Residual Networks as described in https://arxiv.org/abs/1605.07146 with some additional settings and capable of handling image superpositions.

# May be used to remove batch norm by passing it to class constructor
class IdentityNorm(nn.Identity):
    def __init__(self, planes = None):
        super(IdentityNorm, self).__init__()

class BasicBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, norm_layer: Optional[nn.Module] = None, relu_parameter: float =None, skip_init: bool = False) -> None:
        super().__init__()
        r"""Initializes a block of a resnet model
        Args:
            inplanes: number of channels of input tensor
            planes: number of channels of output tensor and intermediate tensor after first convolution
            stride: stride to apply in first convolution
            downsample: None or 1x1 convolution in case direct skip connection is not possible (channel difference)
            norm_layer: normalization layer like BatchNorm
            relu_parameter: initialisation of sReLU/pReLU parameter used in subclasses
            skip_init: whether skipInit is used
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride
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

class SuperWideResnet(nn.Module, Superposition):
    def __init__(self, num_img_sup_cap: int, binding_type: str, width: int, layers: List[int], initial_width: int = 1, num_classes: int = 1000, norm_layer: Optional[nn.Module] = None, block = BasicBlock, relu_parameter = None, skip_init: bool = False, trainable_keys = True, input_channels: int=3) -> None:
        r"""Initialises a Computation in Superposition enabled WideResNet
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
        nn.Module.__init__(self)
        if len(layers)==2: 
            channels = 8*initial_width
            fully_connected_features = 16*width
        elif len(layers)==3: 
            channels = 16*initial_width
            fully_connected_features = 64*width
        elif len(layers)==4: 
            channels = 64*initial_width
            fully_connected_features = 512*width

        Superposition.__init__(self, num_img_sup_cap=num_img_sup_cap, binding_type=binding_type, channels= channels, fully_connected_features= fully_connected_features, trainable_keys = trainable_keys)

        self.skip_init = skip_init
        self.relu_parameter = relu_parameter

        # num_img_sup indicates how many images should be superposed in the forward call, whereas num_img_sup_cap indicates the capabilities of the model. 
        # whereas self.num_img_sup_cap should not be changed after initialisation self.num_img_sup can be adjusted before each forward call.
        self.num_img_sup = num_img_sup_cap # standard behaviour is to use all possible superposition channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if len(layers) == 2:
            self.inplanes = 8*initial_width
            self.conv1 = noOrthoRegularizationConv2d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(block, 8*width, layers[0])
            self.layer2 = self._make_layer(block, 16*width, layers[1], stride=2)
            self.layer3 = nn.Identity()
            self.layer4 = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16 * width, num_classes)
        elif len(layers) == 3:
            self.inplanes = 16*initial_width
            self.conv1 = noOrthoRegularizationConv2d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(block, 16*width, layers[0])
            self.layer2 = self._make_layer(block, 32*width, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64*width, layers[2], stride=2)
            self.layer4 = nn.Identity()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * width, num_classes)
        elif len(layers) == 4:
            self.inplanes = 64*initial_width
            self.conv1 = noOrthoRegularizationConv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64*width, layers[0])
            self.layer2 = self._make_layer(block, 128*width, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256*width, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512*width, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * width, num_classes)           
        else:
            raise Exception("layers must be of length 4 for Imagenet architecture and length 3 for CIFAR, other lengths are disallowed.")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block : Type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        r"""Generates a ResNet layer
        Args:
            block: ResNet block used in layers. Block also admits children of BasicBlock, useful for ISONet type architecture
            planes: number of input channels
            blocks: number of blocks to use in layer
            stride: stride present in first block of layer

        Returns:
            a layer, i.e. sequential of blocks
        """
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, self.relu_parameter, self.skip_init))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, norm_layer, self.relu_parameter, self.skip_init))

        return nn.Sequential(*layers)

    # contrary to isonet not isometry regularization is present
    def isometry_regularization(self, device):
        return 0

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: tensor of size (N, C, H, W)
        Returns:
            a tensor of size (N, number_of_classes)
        """
        assert self.num_img_sup_cap % self.num_img_sup == 0, 'all superposition channels must be used, i.e. the number of superpositions must divide the superposition capacity of the model'
        sup_ratio = self.num_img_sup_cap // self.num_img_sup
        
        eff_batch_size = x.shape[0] - (x.shape[0] % self.num_img_sup)
        x = x[:eff_batch_size, :, :, :] # now batch is divisible by self.num_img_sup

        # Normal WideResNet implementation enabled for processing images in superposition
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Bundling operation after binding images with keys
        assert x.shape[0] % self.num_img_sup == 0
        x = x.reshape(x.shape[0]//self.num_img_sup, self.num_img_sup, x.shape[1], x.shape[2], x.shape[3]) #(N/num_img_sup, num_img_sup, C, H, W)

        # repeat images to be processed together until all available superposition channels are used
        x = x.repeat(1, sup_ratio, 1, 1, 1) #(N/num_img_sup, num_img_sup_cap, C, H, W)

        x = self.bind(x)
        x = torch.sum(x, 1) #(N/num_img_sup, C, H, W)

        # computation in superposition
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) #(N/num_img_sup, C, 1, 1)

        #(N/num_img_sup, C, H, W)
        x = self.unbind(x) #(N/num_img_sup, num_img_sup_cap * C, 1, 1)

        x = x.reshape(eff_batch_size // self.num_img_sup, sup_ratio, self.num_img_sup, self.fc.in_features) #(N/num_img_sup, sup_ratio, num_img_sup , C)
        x = x.mean(dim=1) #(N/num_img_sup, num_img_sup, C) 
                
        x = x.reshape(eff_batch_size, self.fc.in_features) #(N, C) 

        # fully connected layer (linear)
        x = self.fc(x)

        return x
