import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_channels, eps, affine=False)

    def forward(self, input, params):
        ##########################
        weight, bias = params['weight'], params['bias']
        weight.squeeze_()
        bias.squeeze_()
        ##########################

        return F.group_norm(input, self.num_groups, weight, bias, self.eps)


def calc_group_mean_std(feat, num_groups, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    assert (C // num_groups) == (C / num_groups)
    num_channels_per_group = C / num_groups
    feat_var = feat.view(N, num_groups, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, num_groups, 1, 1).repeat(1, num_channels_per_group, 1, 1)
    feat_mean = feat.view(N, num_groups, -1).mean(dim=2)
    feat_mean = feat_mean.view(N, num_groups, 1, 1).repeat(1, num_channels_per_group, 1, 1)
    return feat_mean, feat_std


def adaptive_group_normalization(content_feat, style_params, num_groups):
    '''
    The difference between this function and AdaIn is that AdaIn use style_mean and style_var of NxCx1x1 shape,
    while this function use style_params of 1xCx1x1 shape.
    :param content_feat:
    :param style_params:
    :return:
    '''
    weight, bias = style_params['weight'], style_params['bias']
    size = content_feat.size()
    content_mean, content_std = calc_group_mean_std(content_feat, num_groups)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * weight.expand(size) + bias.expand(size)