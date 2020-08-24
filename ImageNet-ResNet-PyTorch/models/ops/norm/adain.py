import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=False):
        super(AdaptiveInstanceNorm2d, self).__init__(num_features, eps, momentum,
                                                  affine=False, track_running_stats=track_running_stats)

    def forward(self, input, params):
        self._check_input_dim(input)

        ##########################
        weight, bias = params['weight'], params['bias']
        weight.squeeze_()
        bias.squeeze_()
        ##########################

        return F.instance_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


def calc_instance_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]

    # option 1
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    # option 2
    # feat_var = torch.var(feat, dim=(2, 3)).view(N, C, 1, 1)
    # feat_std = torch.std(feat, dim=(2,3)).view(N, C, 1, 1)
    # feat_mean = torch.mean(feat, dim=(2, 3)).view(N, C, 1, 1)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_params):
    '''
    The difference between this function and AdaIn is that AdaIn use style_mean and style_var of NxCx1x1 shape,
    while this function use style_params of 1xCx1x1 shape.
    :param content_feat:
    :param style_params:
    :return:
    '''
    weight, bias = style_params['weight'], style_params['bias']
    size = content_feat.size()
    content_mean, content_std = calc_instance_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * weight.expand(size) + bias.expand(size)
