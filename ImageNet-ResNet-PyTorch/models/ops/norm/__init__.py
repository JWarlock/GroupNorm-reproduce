import torch.nn as nn

from .adabn import AdaptiveBatchNorm2d
from .adagn import AdaptiveGroupNorm
from .adain import AdaptiveInstanceNorm2d
from .frn import FRN

norm_config = {
    # format: layer_type: (abbreviation, module)
    'BN1d': ('bn1d', nn.BatchNorm1d),
    'BN': ('bn', nn.BatchNorm2d),
    'GN': ('gn', nn.GroupNorm),
    'IN': ('in', nn.InstanceNorm2d),
    'FRN': ('frn', FRN),
    'AdaBN': ('adabn', AdaptiveBatchNorm2d),
    'AdaGN': ('adagn', AdaptiveGroupNorm),
    'AdaIN': ('adain', AdaptiveInstanceNorm2d),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_config:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_config[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    # cfg_.setdefault('eps', 1e-5)
    cfg_.setdefault('eps', 1e-6)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer