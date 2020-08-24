import os
from collections import OrderedDict
from itertools import chain
import logging

import torch
from torch import nn as nn
from torch.hub import _get_torch_home


from .ops import Id, build_norm_layer
from .utils import kaiming_init, constant_init


class AlexNetCaffe(nn.Module):
    def __init__(self, mode='classifier', n_classes=1000, dropout=True, norm_config=dict(type='BN')):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        assert mode in ['classifier', 'feature']
        self.mode = mode
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            build_norm_layer(norm_config, 96, postfix=1),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            build_norm_layer(norm_config, 256, postfix=2),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            build_norm_layer(norm_config, 384, postfix=3),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            build_norm_layer(norm_config, 384, postfix=4),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            build_norm_layer(norm_config, 256, postfix=5),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
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

    def forward(self, x, get_feature=False):
        x = self.features(x)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        if self.mode == 'feature' and get_feature:
            return x, self.classifier(x)
        elif self.mode == 'feature':
            return self.classifier(x)
        elif self.mode == 'classifier' and get_feature:
            return x, self.class_classifier(self.classifier(x))
        elif self.mode == 'classifier':
            # import pdb; pdb.set_trace()
            return self.class_classifier(self.classifier(x))
        else:
            raise NotImplementedError

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def caffenet_bn(mode, classes=1000, pretrained=False, model_dir=None):
    model = AlexNetCaffe(mode, classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)

    if pretrained:
        if model_dir is None:
            torch_home = _get_torch_home()
            model_dir = os.path.join(torch_home, 'checkpoints')
        cached_file = os.path.join(model_dir, 'caffenet_checkpoint.pth.tar')
        # optionally resume from a checkpoint
        logging.info("=> loading checkpoint '{}'".format(cached_file))
        checkpoint = torch.load(cached_file, map_location=torch.device('cpu'))
        best_acc1 = checkpoint['best_acc1']
        logging.info(f'best_acc1:{best_acc1}')
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        # load params
        del state_dict["class_classifier.weight"]
        del state_dict["class_classifier.bias"]
        model.load_state_dict(state_dict, strict=False)
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(cached_file, checkpoint['epoch']))
    return model

