import torch
import torchvision
from torch.autograd import Function
import torch.nn as nn
from torch.nn import init
####################

class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x