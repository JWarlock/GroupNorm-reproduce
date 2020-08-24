import torch
import torchvision
from torch.autograd import Function
import torch.nn as nn
from torch.nn import init
####################
# copied from dsn repo
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None
####################
