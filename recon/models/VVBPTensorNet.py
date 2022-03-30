import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
from .RED_CNN import RED_CNN

class bprj_sv_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.backprojection_sv(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        temp = grad_output.sum(1, keepdim=True)
        grad_input = ctlib.backprojection_t(temp.contiguous(), options)
        return grad_input, None

class backprojector_sv(nn.Module):
    def __init__(self):
        super(backprojector_sv, self).__init__()
        
    def forward(self, proj, options):
        return bprj_sv_fun.apply(proj, options)


class FBP_sv(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        dets = int(options[1])
        dDet = options[5]
        s2r = options[7]
        d2r = options[8]
        virdet = dDet * s2r / (s2r + d2r)
        filter = torch.empty(2 * dets - 1)
        pi = torch.acos(torch.tensor(-1.0))
        for i in range(filter.size(0)):
            x = i - dets + 1
            if abs(x) % 2 == 1:
                filter[i] = -1 / (pi * pi * x * x * virdet * virdet)
            elif x == 0:
                filter[i] = 1 / (4 * virdet * virdet)
            else:
                filter[i] = 0
        filter = filter.view(1,1,1,-1)
        w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet)
        w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
        w = w.view(1,1,1,-1) * virdet * pi / options[0]
        self.w = nn.Parameter(w, requires_grad=False)
        self.filter = nn.Parameter(filter, requires_grad=False)
        self.options = nn.Parameter(options, requires_grad=False)
        self.backprojector = backprojector_sv()
        self.dets = dets

    def forward(self, projection):
        p = projection * self.w
        p = torch.nn.functional.conv2d(p, self.filter, padding=(0,self.dets-1))
        recon = self.backprojector(p, self.options)
        return recon

class VVBPTensorNet(nn.Module):
    def __init__(self, options) -> None:
        super().__init__()
        self.backprojector = FBP_sv(options)
        self.model = RED_CNN()
        self.conv = nn.Conv2d(512,1,3,1,1)

    def forward(self, p):
        x = self.backprojector(p)
        x_in, _ = torch.sort(x, dim=1)
        x_in = self.conv(x_in)
        out = self.model(x_in)
        return out