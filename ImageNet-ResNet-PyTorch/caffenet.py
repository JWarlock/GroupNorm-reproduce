import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn
from torch.hub import _get_torch_home


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x

class AlexNetCaffe(nn.Module):
    def __init__(self, mode='classifier', n_classes=1000, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        assert mode in ['classifier', 'feature']
        self.mode = mode
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        if self.mode == 'classifier':
            self.class_classifier = nn.Linear(4096, n_classes)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, domains))

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(),
                                 self.class_classifier.parameters()#, self.domain_classifier.parameters()
                                 ), "lr": base_lr}]

    @property
    def fdim(self):
        return 4096

    def forward(self, x):
        x = self.features(x) # no magic number
        x = x.view(x.size(0), -1)
        if self.mode == 'feature':
            return self.classifier(x)
        elif self.mode == 'classifier':
            x = self.classifier(x)
            return self.class_classifier(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def caffenet(mode, classes=None, pretrained=False, model_dir=None):
    model = AlexNetCaffe(mode, classes)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # if isinstance(m, nn.Linear):
        #     nn.init.xavier_uniform_(m.weight, .1)
        #     nn.init.constant_(m.bias, 0.)
    return model

