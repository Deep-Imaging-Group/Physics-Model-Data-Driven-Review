import torch
import torch.nn as nn
from .LEARN_FBP import FBP

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2d, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 1, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class SubNet(nn.Module):
    def __init__(self, layers):
        super(SubNet, self).__init__()
        self.conv_first = nn.Sequential(nn.Conv2d(1, 64, 5, 1, 2), nn.ReLU(inplace=True))
        self.conv = nn.ModuleList([Conv2d(64, 64) for i in range(layers)])
        self.conv_last = nn.Conv2d(64, 1, 5, 1, 2)
        self.layers = layers

    def forward(self, x):
        y = x.clone()
        y = self.conv_first(y)
        z = y.clone()
        for layer in self.conv:
            y = layer(y)
            z += y
        z = z / (self.layers + 1)
        z = self.conv_last(z)
        out = z + x
        return out


class AdaptiveNet(nn.Module):
    def __init__(self, options):
        super(AdaptiveNet, self).__init__()
        self.model = nn.Sequential(SubNet(3), FBP(options), SubNet(5))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, y):
        out = self.model(y)
        return out
