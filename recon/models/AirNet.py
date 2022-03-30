import torch
import torch.nn as nn
from torch.autograd import Function
from .LEARN import projector
from .LEARN_FBP import fidelity_module

class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, options, idx):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(idx + 1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        self.relu = nn.ReLU(inplace=True)      

    def forward(self, input_data, proj, iter_res):
        mid_res = self.block1(input_data, proj)
        if iter_res is None:
            deep_res = mid_res.clone()
        else:
            deep_res = torch.cat((mid_res, iter_res), dim=1)
        out = self.block2(deep_res) + mid_res
        return out, deep_res

class AirNet(nn.Module):
    def __init__(self, options, block_num=50, hid_channels=48, kernel_size=3, padding=1):
        super(AirNet, self).__init__()
        self.model = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, options, i) for i in range(block_num)])
        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_data, proj):
        x = input_data
        iter_res = None
        for index, module in enumerate(self.model):
            x, iter_res = module(x, proj, iter_res)
        return x