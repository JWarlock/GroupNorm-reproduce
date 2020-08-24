import torch.nn as nn
import torch.nn.functional as F

class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(AdaptiveBatchNorm2d, self).__init__(num_features, eps, momentum,
                                                  affine=False, track_running_stats=track_running_stats)

    def forward(self, input, params):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        ##########################
        weight, bias = params['weight'], params['bias']
        ##########################

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)



def calc_batch_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    C = size[1]
    feat_var = feat.transpose(0,1).view(1, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(1, C, 1, 1)
    feat_mean = feat.transpose(0,1).view(1, C, -1).mean(dim=2).view(1, C, 1, 1)
    return feat_mean, feat_std


def adaptive_batch_normalization2d(content_feat, style_params):
    '''
    The difference between this function and AdaIn is that AdaIn use style_mean and style_var of NxCx1x1 shape,
    while this function use style_params of 1xCx1x1 shape.
    :param content_feat:
    :param style_params:
    :return:
    '''
    weight, bias = style_params['weight'], style_params['bias']
    size = content_feat.size()

    # TODO: Add running_mean/var for testing
    content_mean, content_std = calc_batch_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * weight.expand(size) + bias.expand(size)