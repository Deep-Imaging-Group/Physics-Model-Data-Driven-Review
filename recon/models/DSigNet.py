import torch
import torch.nn as nn
import math
import numpy as np 
from torch.autograd import Function
from .iRadonMap import PixelIndexCal_cuda
from .iRadonMap import BackProjNet

def PixelIndexCal_DownSampling(length, width, lds, wds):
    length, width = int(length/lds), int(width/wds)
    ds_indices = torch.zeros(lds*wds, width*length).type(torch.LongTensor)
    for x in range(lds):
        for y in range(wds):
            k = x*width*wds+y
            for z in range(length):
                i, j = z*width, x*wds+y
                st = k+z*width*wds*lds
                ds_indices[j, i:i+width] = torch.tensor(range(st,st+width*wds, wds))
    return ds_indices.view(-1)


def PixelIndexCal_UpSampling(index, length, width):
    index = index.view(-1)
    _, ups_indices = index.sort(dim=0, descending=False)
    return ups_indices.view(-1)        

class DownSamplingBlock(nn.Module):
    def __init__(self, planes=8, length=512, width=736, lds=2, wds=2):
        super(DownSamplingBlock, self).__init__()
        self.length = int(length/lds)
        self.width = int(width/wds)
        self.extra_channel = lds*wds
        self.ds_index = nn.Parameter(PixelIndexCal_DownSampling(length, width, lds, wds), requires_grad=False)
        self.filter = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        _, channel, length, width = input.size()
        output = torch.index_select(input.view(-1, channel, length*width), 2, self.ds_index)
        output = output.view(-1, channel*self.extra_channel, self.length, self.width)
        output = self.leakyrelu(self.ln(self.filter(output)))

        return output


class UpSamplingBlock(nn.Module):
    def __init__(self, planes=8, length=64, width=64, lups=2, wups=2):
        super(UpSamplingBlock, self).__init__()

        self.length = length*lups
        self.width = width*wups
        self.extra_channel = lups*wups
        ds_index = PixelIndexCal_DownSampling(self.length, self.width, lups, wups)
        self.ups_index = nn.Parameter(PixelIndexCal_UpSampling(ds_index, self.length, self.width), requires_grad=False)
        self.filter = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        
    def forward(self, input):
        _, channel, length, width = input.size()
        channel = int(channel/self.extra_channel)
        output = torch.index_select(input.view(-1, channel, self.extra_channel*length*width), 2, self.ups_index)
        output = output.view(-1, channel, self.length, self.width)
        output = self.leakyrelu(self.ln(self.filter(output)))

        return output


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()

        self.filter1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output += input
        output = self.leakyrelu2(output)

        return output


class SinoNet(nn.Module):
    def __init__(self, bp_channel, num_filters):
        super(SinoNet, self).__init__()

        model_list  = [nn.Conv2d(1, num_filters, kernel_size=3, stride=1, padding=1, bias=True), nn.GroupNorm(num_channels=num_filters, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        model_list += [DownSamplingBlock(planes=num_filters*4, length=512, width=736, lds=2, wds=2)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        
        model_list += [nn.Conv2d(num_filters*4, bp_channel, kernel_size=1, stride=1, padding=0, bias=True)]
        self.model = nn.Sequential(*model_list)

    def forward(self, input):

        return self.model(input)

class SpatialNet(nn.Module):
    def __init__(self, bp_channel, num_filters):
        super(SpatialNet, self).__init__()

        model_list  = [nn.Conv2d(bp_channel, num_filters*4, kernel_size=3, stride=1, padding=1, bias=True), nn.GroupNorm(num_channels=num_filters*4, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4), ResidualBlock(planes=num_filters*4)]
        model_list += [UpSamplingBlock(planes=num_filters, length=256, width=256, lups=2, wups=2)]
        
        model_list += [nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True)]
        self.model = nn.Sequential(*model_list)

    def forward(self, input):

        return self.model(input)

class DSigNet(nn.Module):
    def __init__(self, options, bp_channel=4, num_filters=16, scale_factor=2) -> None:
        super().__init__()
        geo_real = {'nVoxelX': int(options[2]), 'sVoxelX': float(options[4]) * int(options[2]), 'dVoxelX': float(options[4]), 
            'nVoxelY': int(options[3]), 'sVoxelY': float(options[4]) * int(options[3]), 'dVoxelY': float(options[4]), 
            'nDetecU': int(options[1]), 'sDetecU': float(options[5]) * int(options[1]), 'dDetecU': float(options[5]), 
            'offOriginX': 0.0, 'offOriginY': 0.0, 
            'views': int(options[0]), 'slices': 1,
            'DSD': float(options[8]) + float(options[9]), 'DSO': float(options[8]), 'DOD': float(options[9]),
            'start_angle': 0.0, 'end_angle': float(options[7]) * int(options[0]),
            'mode': 'fanflat', 'extent': 1, # currently extent supports 1, 2, or 3.
            }
        geo_virtual = dict()
        geo_virtual.update({x: int(geo_real[x]/scale_factor) for x in ['views']})
        geo_virtual.update({x: int(geo_real[x]/scale_factor) for x in ['nVoxelX', 'nVoxelY', 'nDetecU']})
        geo_virtual.update({x: geo_real[x]/scale_factor for x in ['sVoxelX', 'sVoxelY', 'sDetecU', 'DSD', 'DSO', 'DOD', 'offOriginX', 'offOriginY']})
        geo_virtual.update({x: geo_real[x] for x in ['dVoxelX', 'dVoxelY', 'dDetecU', 'slices', 'start_angle', 'end_angle', 'mode', 'extent']})
        geo_virtual['indices'], geo_virtual['weights'] = PixelIndexCal_cuda(geo_virtual)
        self.SinoNet = SinoNet(bp_channel, num_filters)
        self.BackProjNet = BackProjNet(geo_virtual, bp_channel)
        self.SpatialNet = SpatialNet(bp_channel, num_filters)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input):
        output = self.SinoNet(input)
        output = self.BackProjNet(output)
        output = self.SpatialNet(output)
        return output

