import torch.nn as nn
from .tlu import TLU


activation_config = {
    # format: layer_type: (abbreviation, module)
    # ReLU usually comes with replace=True
    'ReLU': ('relu', nn.ReLU),
    'TLU': ('tlu', TLU),
    # and potentially 'SN'
}


def build_activation(cfg, num_features=None, postfix=''):
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
    if layer_type not in activation_config:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, activation_layer = activation_config[layer_type]
        if activation_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    if num_features is not None:
        cfg_.update(num_features=num_features)
    if layer_type == 'TLU':
        assert 'num_features' in cfg_

    # import pdb; pdb.set_trace()

    layer = activation_layer(**cfg_)

    requires_grad = cfg_.pop('requires_grad', True)
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer