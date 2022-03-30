import torch
import torch.nn as nn
from torch.autograd import Function
import math
import torchvision.transforms as transforms

class rotation(nn.Module):
    def __init__(self, dAng, height, width):
        super(rotation, self).__init__()
        self.dAng = dAng * 180 / math.pi
        self.height = height
        self.width = width

    def forward(self, x):
        B, N2, Nv, L = x.shape
        y = x.transpose(1, 3).view(B, Nv, self.height, self.width).contiguous()
        out = torch.empty_like(y)
        for i in range(Nv):
            ang = self.dAng * i
            out[:, i] = transforms.functional.rotate(y[:, i], ang, transforms.InterpolationMode.BILINEAR)
        out = out.view(B, Nv, N2, 1).contiguous()
        return out


class iCTNet(nn.Module):
    def __init__(self, N_v, N_c, height, width, dAng, beta=5, alpha_1 = 1, alpha_2 = 1):
        super(iCTNet, self).__init__()
        self.L1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3), padding=(0, 1)),
            nn.Hardshrink(1e-5)
        )
        self.L2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), padding=(0, 1)),
            nn.Hardshrink(1e-5)
        )
        self.L3 = nn.Sequential(
            nn.Conv2d(129, 1, (1, 3), padding=(0, 1)),
            nn.Hardshrink(1e-5)
        )
        self.L4 = nn.Sequential(
            nn.Conv2d(N_v, N_v * alpha_1, (1, 1), padding=(0, 0)),
            nn.Hardshrink(1e-8)
        )
        self.L5 = nn.Sequential(
            nn.Conv2d(N_v * alpha_1, N_v * alpha_2, (1, 1), padding=(0, 0)),
            nn.Hardshrink(1e-8)
        )
        self.L6 = nn.Sequential(
            nn.Conv2d(1, 1, (N_v * alpha_2, N_c), padding='same', padding_mode='circular', bias=False),
            nn.Identity()
        )
        self.L7 = nn.Sequential(
            nn.Conv2d(1, 16, (1, beta), padding=(0 ,(beta-1)//2)),
            nn.Tanh()
        )
        self.L8 = nn.Sequential(
            nn.Conv2d(16, 1, (1, beta), padding=(0, (beta-1)//2)),
            nn.Tanh()
        )
        self.L9 = nn.Sequential(
            nn.Conv2d(N_c, N_c, (1, 1), padding=(0, 0)),
            nn.Tanh()
        )
        self.L10 = nn.Sequential(
            nn.Conv2d(N_c, height * width, (1, 1), padding=(0, 0), bias=False),
            nn.Identity()
        )
        self.L11 = rotation(dAng, height, width)
        self.L12 = nn.Sequential(
            nn.Conv2d(N_v * alpha_2, 1, (1, 1), padding=(0, 0), bias=False),
            nn.Identity()
        )
        self.height = height
        self.width = width
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x1 = self.L1(x)
        x2 = self.L2(x1)
        x3_in = torch.cat((x,x1,x2), dim=1)
        x3 = self.L3(x3_in)
        x4_in = x3.transpose(1, 2)
        x4 = self.L4(x4_in)
        x5 = self.L5(x4)
        x6_in = x5.transpose(1, 2)
        x6 =self.L6(x6_in)
        x7 = self.L7(x6)
        x8 = self.L8(x7)
        x9_in = x8.transpose(1, 3)
        x9 = self.L9(x9_in)
        x10 = self.L10(x9)
        x11 = self.L11(x10)
        x12 = self.L12(x11)
        out = x12.view(x12.size(0), 1, self.height, self.width)
        return out

    def segment1(self, x):
        x1 = self.L1(x)
        x2 = self.L2(x1)
        x3_in = torch.cat((x,x1,x2), dim=1)
        out = self.L3(x3_in)
        return out
    
    def segment2(self, x):
        x4_in = x.transpose(1, 2)
        x4 = self.L4(x4_in)
        x5 = self.L5(x4)
        out = x5.transpose(1, 2)
        return out

    def segment3(self, x):
        x6 =self.L6(x)
        x7 = self.L7(x6)
        x8 = self.L8(x7)
        x9_in = x8.transpose(1, 3)
        x9 = self.L9(x9_in)
        out = x9.transpose(1, 3)
        return out

    def segment4(self, x):
        x = x.transpose(1, 3)
        x10 = self.L10(x)
        x11 = self.L11(x10)
        x12 = self.L12(x11)
        out = x12.view(x12.size(0), 1, self.height, self.width)
        return out
