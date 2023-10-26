#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
from torch.optim import SGD
from models.superwideisonet import SReLU 
from torch.nn import PReLU

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
def construct_optim(net, weight_decay:float):
    r"""construct custom optimizer to not apply weight decay to bias parameter in sReLU and slope parameter in pReLU. Learning rate and momentum will be handled by scheduler policy.
    Args:
        net: torch model
        weight_decay: strength of weight decay
    Returns:
        SGD optimizer with the given weight decay only applied to non-sReLU/pReLU parameters
    """
    relu_params = []
    other_params = []
    for m in net.modules():
        if isinstance(m, SReLU) or isinstance(m, PReLU):
            relu_params.extend(m.parameters(recurse=False))
        else:
            other_params.extend(m.parameters(recurse=False))

    optim_params = [
        {
            'params': relu_params,
            'weight_decay': 0,
        },
        {
            'params': other_params,
            'weight_decay': weight_decay,
        }
    ]

    # Check all parameters are accounted for.
    assert len(list(net.parameters())) == len(other_params) + len(relu_params), \
        f'parameter size does not match: ' \
        f'{len(other_params)} + {len(relu_params)} != ' \
        f'{len(list(net.parameters()))}'

    return SGD(
        optim_params,
        lr=0,
        momentum=0,
        weight_decay=weight_decay,
        dampening=0,
        nesterov=False
    )