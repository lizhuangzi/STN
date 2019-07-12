import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import OrderedDict

class SpatialCom(nn.Module):
    def __init__(self):
        super(SpatialCom, self).__init__()
        self.primaryConv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)),
        ]))
        self.primaryConv.add_module('prelu0', nn.PReLU())

        layers = []
        for i in range(19):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.PReLU(),
            )
            layers.append(layer)

        self.sptioCNN = nn.Sequential(*layers)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU()
        )

        self.recon = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        x = self.primaryConv(x)
        x = self.sptioCNN(x)
        x = self.deconv(x)
        return self.recon(x)
